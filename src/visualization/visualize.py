import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from src.models.predict_model import *

def get_class_plot_prop():
    """
    Returns dict specifying colors and other graphics properties 
    to be used for plotting. Specifically:
    color name for line plots, colormap for images, RGB color vector for individual pixels
    """
    CLASS_PLOT_PROP = {
        "no_img": ["black", "gray", np.array([0, 0, 0], dtype=np.uint8).reshape(1, 1, 3)],
        "no_road": ["gray", "gray", np.array([40, 40, 40], dtype=np.uint8).reshape(1, 1, 3)],
        "paved_road": ["blue", "bone", np.array([120, 120, 255], dtype=np.uint8).reshape(1, 1, 3)],
        "any_road": ["black", "gnuplot", np.array([200, 200, 200], dtype=np.uint8).reshape(1, 1, 3)],
        "unpaved_road": ["darkorange", "copper", np.array([200, 180, 10], dtype=np.uint8).reshape(1, 1, 3)],
    }
    return CLASS_PLOT_PROP


def grayscale_to_rgb(x, class_plot_prop, class_dict):
    """
    Converts a 2D array representing road labels into a 3D array such that when
    it is imshown any legal labels appear in colors defined by class_plot_prop
    """
    assert(x.ndim >= 2), "x must be at least 2D"
    if x.ndim == 3:
        if x.shape[2] == 3:
            print("x is 3D already")
            rgb = x
        elif x.shape[2] == 1:
            rgb = np.tile(x.copy(), [1, 1, 3])
        else:
            # give up
            raise Exception("x is 3D with two layers")
    else:
        rgb = np.tile(x.copy().reshape(x.shape +(1,)), [1, 1, 3])
    inverted_class_dict = {value: key for key, value in zip(class_dict.keys(), class_dict.values())}
    unique_vals = np.unique(x)
    # loop thru unique values
    for uix in set(unique_vals).intersection(set(inverted_class_dict.keys())):
        map = x.reshape(x.shape[:2]) == uix
        rgb[map,:] = class_plot_prop[inverted_class_dict[uix]][2]
    return rgb


def show_tile(tile, ax, cmap=None, scale=None, show_colorbar=False, title=None, **kwargs):
    """
    Custom wrapper for imshow tailored to satellite image tiles and numpy
    arrays derived from them
    scale must be specified as pixels per meter
    """
    im_h = None
    if ax:
        # get rid of singleton 3rd dimension
        if tile.ndim >=3:
            if tile.shape[2] == 1:
                tile = tile.reshape(tile.shape[:2])
        im_h = ax.imshow(tile, cmap=cmap, **kwargs);
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        if scale:
            print('plotting scale')
            # depict a line representing 200 m in lower left 
            x_span = 200.0 * scale
            x_co = np.array(tile.shape[1] * 0.05) + np.array([0, x_span])
            y_co = np.array(tile.shape[0] * 0.95) + np.zeros((2))
            ax.plot(x_co, y_co, color = 'white', linewidth = 3)            
        if show_colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im_h, cax=cax)
        ax.set(title=title)
    return im_h


def plot_pr(recall_dict, precision_dict, auc_pr_dict, beven_ix_dict, beven_thresh_dict, ax, plot_prop=get_class_plot_prop()):
    """
    Generates precision-recall curve plot with embellishments. Downsamples
    inputs recall and precision for plot to between 5000 and 10000 points.
    """
    if len(recall_dict):
        for i, k in enumerate(recall_dict.keys()):
            # downsampling for plot:
            ix = (len(recall_dict[k]) - np.logspace(0, np.log10(len(recall_dict[k])), num=2000)).astype(np.int64)
            ix = ix[::-1]
            recall = recall_dict[k][ix]
            precision = precision_dict[k][ix]
            # plot curve
            ax.plot(recall, precision, \
                    label="{0:s}: auc = {1:0.2f}".format(str(k), auc_pr_dict[k]),
                    color = plot_prop[k][0])
            if ((beven_ix_dict is not None) and (beven_thresh_dict is not None)):
                ax.plot(np.r_[0, recall_dict[k][beven_ix_dict[k]]], precision_dict[k][beven_ix_dict[k]]*np.array([1, 1],dtype=int),
                       linestyle=':', color='gray')
                ax.plot(recall_dict[k][beven_ix_dict[k]]*np.array([1, 1],dtype=int), np.r_[0, precision_dict[k][beven_ix_dict[k]]],
                       linestyle=':', color='gray')
                ax.text(0.1, 0.9 - i*0.1, 
                        "break-even {0:0.2f} @thresh {1:0.2f}".format(recall_dict[k][beven_ix_dict[k]], beven_thresh_dict[k]))
        ax.set_yticks(np.arange(0, 1.25, 0.25))
        ax.set_xticks(np.arange(0, 1.25, 0.25))
        ax.set_ylim(-0.01, 1.01)
        ax.set_xlim(-0.01, 1.01)
        ax.legend(loc="lower right")
        ax.set(title='PR', xlabel='recall', ylabel='precision')


def plot_roc(fpr_dict, tpr_dict, auc_roc_dict, ax, plot_prop=get_class_plot_prop()):
    """
    Generates receiver-operating characteristic (roc) curve plot with embellishments. 
    Downsamples inputs recall and precision for plot to between 5000 and 10000 points.
    """    
    if len(fpr_dict):
        # plot diagonal
        ax.plot(np.linspace(0, 1), np.linspace(0, 1), linestyle='--', color='gray')
        for k in fpr_dict.keys():
            # determine downsampling factor for plot:
            ds_fac = np.max([len(fpr_dict[k]) // 5e3, 1]).astype(int)
            # downsampling:
            fpr = fpr_dict[k][::ds_fac]
            tpr = tpr_dict[k][::ds_fac]
            ax.plot(fpr, tpr,
                    label="{0:s}: auc = {1:0.2f}".format(str(k), auc_roc_dict[k]),
                    color = plot_prop[k][0])
        ax.set_yticks(np.arange(0, 1.25, 0.25))
        ax.set_xticks(np.arange(0, 1.25, 0.25))
        ax.set_ylim(-0.01, 1.01)
        ax.set_xlim(-0.01, 1.01)
        ax.legend(loc="lower right")
        ax.set(title='ROC', xlabel='false positive rate', ylabel='true positive rate')


def show_sample_prediction(x, y, yscore, ypred, class_dict, scale=None, title=None, display_mode="full"):
    """
    Produces a multipanel plot of an individual sample (image tile), its
    label, its prediction and analytics
    """
    assert(display_mode in ["full", "compact"]), "display mode must be either of 'compact' and 'full'"
    img_size = np.prod(y.shape[:2])
    dim_yscore = yscore.shape[2]
    if dim_yscore > 1:
        type_model = "multiclass"
    else:
        type_model = "binary"
    
    if display_mode == "full":
        sp_r, sp_c = 2, 4
        fig_sample, axs = plt.subplots(sp_r, sp_c, figsize=(12, 6))
        ax_rgb = axs[0,0]
        ax_nir = axs[0,1]
        ax_yscore_binary = axs[1,0]
        ax_yscore_rgb_binary = axs[1,1]
        ax_yscore_paved = axs[1,0]
        ax_yscore_unpaved = axs[1,1]
        ax_ylabel = axs[0,2]
        ax_ylabel_pred = axs[0,3]
    else:
        sp_r, sp_c = 1, 4 + int(type_model == "multiclass")
        fig_sample, axs = plt.subplots(sp_r, sp_c, figsize=(12, 3))
        ax_rgb = axs[0]
        ax_nir = None
        ax_yscore_binary = None
        ax_yscore_rgb_binary = axs[2]
        ax_yscore_paved = axs[2]
        ax_yscore_unpaved = axs[3]
        ax_ylabel = axs[1]
        ax_ylabel_pred = axs[4]
        
    # retrieve plot properties of different classes
    class_plot_prop = get_class_plot_prop()
    # reshaped versions for metric
    y_reshaped = y.reshape((img_size, 1), order = 'C')
    yscore_reshaped = yscore.reshape((img_size, dim_yscore), order = 'C')
    # versions for display in which we get rid of upper small percentile 
    yscore_plot = np.copy(yscore)
    prc = np.percentile(yscore_plot, [99.9])
    yscore_plot[yscore_plot >= prc] = prc
    # ----------------plot ---------------
    fig_sample.suptitle(title, fontsize=16)
    # plot rgb part of image
    show_tile(x[:,:,[2, 1, 0]], ax_rgb, title="RGB", scale=scale);
    # nir
    show_tile(x[:,:,3], ax_nir, cmap="gray",  title="infrared");
    # y score (prediction)
    if type_model == "binary":
        cmap = class_plot_prop["any_road"][1]
        show_tile(yscore_plot, ax_yscore_binary, cmap=cmap, show_colorbar=True,  title="prediction");
        # pale rgb + transparent prediction
        show_tile(exposure.adjust_gamma(x[:,:,[2, 1, 0]], 0.5), ax_yscore_rgb_binary);
        show_tile(yscore_plot, ax_yscore_rgb_binary, cmap=cmap, title="rgb + prediction", alpha=.5);
    else:
        cmap = class_plot_prop["paved_road"][1]
        # assume that the first layer represents scores for no_road
        show_tile(yscore_plot[:,:,1], ax_yscore_paved, cmap=cmap, show_colorbar=True, title="pred. (paved)");
        cmap = class_plot_prop["unpaved_road"][1]
        show_tile(yscore_plot[:,:,2], ax_yscore_unpaved, cmap=cmap, show_colorbar=True, title="prediction (unpaved");
        # convert true labels to rgb
        y = grayscale_to_rgb(y, class_plot_prop, class_dict)
        # convert predicted labels to rgb
        ypred = grayscale_to_rgb(ypred, class_plot_prop, class_dict)

    # true labels (cmap will be ignored if y is rgb)
    show_tile(y, ax_ylabel, cmap="gray", title="true labels");
    # predicted labels
    show_tile(ypred, ax_ylabel_pred, cmap="gray", title="predicted labels");
    if display_mode == "full":
        # auc_roc, auc_pr
        (fpr_sample_dict,
        tpr_sample_dict,
        roc_auc_sample_dict,
        precision_sample_dict,
        recall_sample_dict,
        pr_auc_sample_dict,
        _, _,
        reduced_label_sample_dict) = multiclass_roc_pr(y_reshaped, yscore_reshaped, class_dict=class_dict)
        plot_pr(recall_sample_dict, precision_sample_dict, pr_auc_sample_dict, None, None, axs[1, 2])
        plot_roc(fpr_sample_dict, tpr_sample_dict, roc_auc_sample_dict, axs[1, 3])
    
    return fig_sample