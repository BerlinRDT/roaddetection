import numpy as np
from sklearn import metrics
from sklearn.preprocessing import label_binarize

def get_class_dict(type_dict="all_legal"):
    """
    Defines values of different pixel classes (paved road, no_road, etc.) for
    the purpose of model analysis.
    *** IMPORTANT: *** 
    - the values assigned to paved_road and unpaved_road MUST be equal to the
      corresponding values used during model training
    - during training, no distinction had been made between pixels not corresponding
      to roads and pixels not corresponding to any part of the underlying satellite
      image. That distinction is here made, by reassigning no_road a value of 40
      and no_img a value of zero
    - any_road is another new class necessary for analysis and display of the data;
      its value is arbitrary except for the condition that follows:
    - any road type of interest must have a larger value than no_img and no_road
    See refactor_labels below
    """
    assert(type_dict in ["all_legal", "binary", "multiclass"])
    if type_dict == "all_legal":
        class_dict = {
            "no_img": 0,
            "no_road": 40,
            "paved_road": 127,
            "any_road": 200,
            "unpaved_road": 255
        }
    elif type_dict == "binary":
        class_dict = {
            "no_img": 0,
            "no_road": 40,
            "any_road": 200,
        }
    else:
        class_dict = {
            "no_img": 0,
            "no_road": 40,
            "paved_road": 127,
            "unpaved_road": 255
        }
    return class_dict


def get_sorted_key_index(class_key, class_dict):
    """
    Given class_dict, returns an index that would result from ordering the keys 
    in class_dict according to their values and accessing one of the keys (class_key).
    That index can be used to access multiclass prediction scores
    """
    assert(isinstance(class_key, str))
    keys_sorted = list(sorted(class_dict, key=class_dict.__getitem__, reverse=False))
    ix = [i for i, k in enumerate(keys_sorted) if k == class_key]
    if not len(ix):
        raise Exception("key not found in class_dict")
    return ix[0]


def predict_labels(yscore, thresh_dict, class_dict):
    local_thresh_dict = thresh_dict.copy()
    model_is_binary = not ((yscore.ndim == 3) and (yscore.shape[2] > 1))
    # for the purpose of generating muticlass predicted labels the union of all
    # roads, termed "any_road" (as in the binary case), is not helpful at all,
    # so get rid of it
    if not model_is_binary:
        del local_thresh_dict["any_road"]
    ypred = np.ones(yscore.shape[:2] + (1,), dtype=np.uint8) * class_dict["no_road"]
    for i, k in enumerate(local_thresh_dict):
        if model_is_binary:
            ix = 0
        else:
            ix = get_sorted_key_index(k, class_dict)
        above = yscore[:,:,ix] >= thresh_dict[k]
        ypred[above] = class_dict[k]
    return ypred
        
    
def refactor_labels(x, y, class_dict, model_is_binary=True, meta=None):
    """
    Returns label array y which will be modified according to these rules:
    - if model_is_binary is True, any label with a value above that of 'no_road' 
      will be converted to 'any_road'
    - pixels which are outside original image bounds are converted to 'no_img'
    Also returns mask, alogical array indicatng no_img pixels
    """
    # determine invalid pixels (for now defined as those with a vale of zero
    # in the first band). Variable mask could be used to create a masked array,
    # but scikit-learn does not support masked arrays
    if ((x.dtype == np.float64) or (x.dtype == np.float32)):
        raise Exception("img must be a uint for label refactoring")
    mask = x[:,:,0] == 0;
    # set masked values: first, set zeros in label file to 'no road' value...
    y[np.logical_and(np.logical_not(mask), np.logical_not(y))] = class_dict["no_road"]
    # then set pixel positions found to not belong to image to 'no_img' value
    y[mask] = class_dict["no_img"]
    # if the model used for prediction is a binary one, set all road categories
    # to "any_road"
    if model_is_binary:
        y[y == class_dict["paved_road"]] = class_dict["any_road"]
        y[y == class_dict["unpaved_road"]] = class_dict["any_road"]
    return y, mask


def multiclass_roc_pr(y, yscore, class_dict=get_class_dict()):
    """
    Perform binary or multi-class roc and pr analyses in which 
    - data of the "no_img" class are excluded 
    - all road classes are individually compared to the rest
    - the union of road classes is compared to the "no_road" class
    With n being the number of samples (pixels) and num_classes the number of
    classes represented in yscore:
    y - n by 1 array of labels
    yscore - n by num_class array of prediction scores
    class_dict - dictionary listing all legal values in y
    """
    debug_mode = False
    # - instantiate
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    precision = dict()
    recall = dict()
    pr_auc = dict() 
    beven_ix = dict()
    beven_thresh = dict()
    reduced_class_dict = dict()
    # sanity check: make sure all labels in y actually exist in class_dict:
    unique_labels = np.unique(y)
    assert(not set(unique_labels).difference(set(class_dict.values()))), "illegal label"
    #--------------------------------------------------------------------------
    # only if any class other than no_img and no_road are present do we proceed
    #--------------------------------------------------------------------------
    if len(set(unique_labels).difference((class_dict["no_img"], class_dict["no_road"]))):
        # get rid of singleton 2nd dimension in y
        if y.ndim >=2:
            if y.shape[1] == 1:
                y = y.ravel()
            else:
                raise Exception("2nd dim in y is not singleton")
        # due to Python's indexing logic, make sure that yscore has a 2nd dim,
        # no matter whether it is singleton
        if yscore.ndim <= 1:
            yscore = yscore.reshape(yscore.shape + (1,))
        # now determine number of classes in model based on columns in yscore
        num_class = max(2, yscore.shape[1])
        
        # remove all entries corresponding to no_img because they are pointless
        good_ix = (y != class_dict["no_img"])
        if debug_mode:
            print("excluding {0:0.0f} % non-image pixels)...".format(100*(1.0 - np.sum(good_ix)/y.size)))
            print(y.shape, good_ix.shape, yscore.shape)
        y = y[good_ix]
        yscore = yscore[good_ix,:]
        # remove no_img key
        reduced_class_dict = class_dict.copy()
        del reduced_class_dict["no_img"]
        
        # interim check
        if ((num_class == 2) and (yscore.shape[1] > 1)):
            raise Exception("in binary classification, a single-column yscore is expected")

        # keys for fpr, tpr and roc_auc, indicating the classes to be tested against all others
        keys = [k for k in reduced_class_dict.keys() if k != "no_road"]
        if debug_mode:
            print(reduced_class_dict, keys)
        # binarize on all categories found in y
        # §§§§§§ ensure order
        y_multilabel = label_binarize(y, list(reduced_class_dict.values()))
        # if it's a binary problem, y_multilabel is [nsamples x 1], so we have to reshape
        if num_class == 2:
            y_multilabel = y_multilabel.reshape((-1,1))
        union_ix = []
        for i, k in enumerate(keys):
            if num_class == 2:
                ix = i
            else:
                ix = get_sorted_key_index(k, reduced_class_dict)
            union_ix.append(ix)
            fpr[k], tpr[k], _ = metrics.roc_curve(y_multilabel[:,ix].ravel(), yscore[:,ix].ravel())
            roc_auc[k] = metrics.auc(fpr[k], tpr[k])
            # precision-recall and its auc
            precision[k], recall[k], thresholds = \
                    metrics.precision_recall_curve(y_multilabel[:,ix].ravel(), yscore[:,ix].ravel())
            pr_auc[k] = metrics.auc(recall[k].reshape(-1,), precision[k].reshape(-1,))
            # breakeven point (threshold at which recall and precision are identical)
            beven_ix[k] = np.argmin(np.abs(precision[k] - recall[k]))
            beven_thresh[k] = thresholds[beven_ix[k]]
            # diagnostics (to be removed in the future)
            if np.mean(np.diff(precision[k][::10])) <= 0.0:
                print("WARNING: precision values should be ascending")
            if np.mean(np.diff(recall[k][::10])) >= 0.0:
                print("WARNING: recall values should be descending")
         
        if num_class > 2:
            # compute union of roads vs no_road
            k = "any_road"
            fpr[k], tpr[k], _ = metrics.roc_curve(y_multilabel[:,union_ix].ravel(), yscore[:,union_ix].ravel())
            roc_auc[k] = metrics.auc(fpr[k], tpr[k])
            # precision-recall and its auc
            precision[k], recall[k], thresholds = \
                    metrics.precision_recall_curve(y_multilabel[:,union_ix].ravel(), yscore[:,union_ix].ravel())
            pr_auc[k] = metrics.auc(recall[k], precision[k])
            # breakeven point (threshold at which recall and precision are identical)
            beven_ix[k] = np.argmin(np.abs(precision[k] - recall[k]))
            beven_thresh[k] = thresholds[beven_ix[k]]            
    else:
        print("skipping computations due to absence of labels of interest" )
    return fpr, tpr, roc_auc, precision, recall, pr_auc, beven_ix, beven_thresh, reduced_class_dict