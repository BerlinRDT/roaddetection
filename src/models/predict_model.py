import numpy as np
from sklearn import metrics
from sklearn.preprocessing import label_binarize

def get_class_dict():
    """
    Defines values of different pixel classes (paved road, no_road, etc.) for
    the purpose of model analysis.
    IMPORTANT: ** any road type of interest must have a larger value 
    than no_img and no_road **
    """
    CLASS_DICT = {
        "no_img": 0,
        "no_road": 40,
        "paved_road": 127,
        "any_road": 200,
        "unpaved_road": 255
    }
    return CLASS_DICT

def multiclass_roc_pr(y, yscore, class_dict=get_class_dict()):
    """
    Perform binary or multi-class roc and pr analyses in which all classes are compared
    to the "no_road" class and data of the "no_img" class are ignored
    y - 1D array of labels
    yscore - array of prediction scores
    class_dict - dictionary listing all legal values in y
    """
    debug_mode = False
    # - allocate
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
        # remove all entries corresponding to no_img because they are pointless
        good_ix = y != class_dict["no_img"]
        if debug_mode:
            print("excluding {0:0.0f} % non-image pixels)...".format(100*(1.0 - np.sum(good_ix)/y.size)))
            print(good_ix)
        y = y[good_ix]
        yscore = yscore[good_ix]
        if debug_mode:
            print(y)    
        # consider only labels which exist in y
        unique_labels = np.unique(y)
        reduced_class_dict = {list(class_dict.keys())[i]:list(class_dict.values())[i] \
                      for i, val in enumerate(list(class_dict.values())) if val in unique_labels}
        num_label = len(reduced_class_dict)
        if debug_mode:
            print(unique_labels, reduced_class_dict)


        # to do: remove corresponding layers (?) in yscore    
        if yscore.ndim >1:
            raise Exception("multi-label scores not yet implemented")

        # binarize on all categories found in y
        y_multilabel = label_binarize(y, list(reduced_class_dict.values()))
        # Compute ROC curve and ROC area for each non-no_road class:

        # keys for fpr, tpr and roc_auc, indicating the class which is tested against no_road
        keys = [k for k in reduced_class_dict.keys() if k != "no_road"]
        # if it's a binary problem, y_multilabel is [nsamples x 1]
        if num_label == 2:
            # roc and its auc
            fpr[keys[0]], tpr[keys[0]], _ = metrics.roc_curve(y_multilabel, yscore)
            roc_auc[keys[0]] = metrics.auc(fpr[keys[0]], tpr[keys[0]])
            # precision-recall and its auc
            precision[keys[0]], recall[keys[0]], thresholds = \
                    metrics.precision_recall_curve(y_multilabel, yscore)
            pr_auc[keys[0]] = metrics.auc(recall[keys[0]], precision[keys[0]])
            # breakeven point (threshold at which recall and precision are identical)
            beven_ix[keys[0]] = np.argmin(np.abs(precision[keys[0]] - recall[keys[0]]))
            beven_thresh[keys[0]] = thresholds[beven_ix[keys[0]]]

        elif num_label >= 3:
            raise Exception("multilabel computation of auroc not yet implemented")
            for i in range(1, num_label):
                pass
            # assign averages  
            #fpr["avg"], tpr["avg"], roc_auc["avg"] = fpr[keys[0]], tpr[keys[0]], roc_auc[keys[0]]
    else:
        print("skipping computations due to absence of labels of interest" )
    return fpr, tpr, roc_auc, precision, recall, pr_auc, beven_ix, beven_thresh, reduced_class_dict