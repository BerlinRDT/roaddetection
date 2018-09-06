import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from src.models.predict_model import *

def get_class_plot_prop():
    """
    Returns dict specifying colors and other graphics properties 
    to be used for plotting. Specifically:
    color for line plots, colormap for images
    """
    CLASS_PLOT_PROP = {
        "no_road": ["gray", "gray"],
        "paved_road": ["navy", "bone"],
        "any_road": ["blue", "gnuplot"],
        "unpaved_road": ["darkorange", "copper"],
    }
    return CLASS_PLOT_PROP


def show_tile(tile, ax, cmap=None, show_colorbar=False, title=None, **kwargs):
    """
    Custom wrapper for imshow tailored to satellite image tiles and numpy
    arrays derived from them
    """
    # get rid of singleton 3rd dimension
    if tile.ndim >=3:
        if tile.shape[2] == 1:
            tile = tile.reshape(tile.shape[:2])
    im_h = ax.imshow(tile, cmap=cmap, **kwargs);
    ax.set_yticklabels([])
    ax.set_xticklabels([])
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
            # determine downsampling factor for plot:
            ds_fac = np.max([len(recall_dict[k]) // 5e3, 1]).astype(int)
            # downsampling:
            recall = recall_dict[k][::ds_fac]
            precision = precision_dict[k][::ds_fac]
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


def show_sample_prediction(x, y, yscore, class_dict, title=None):
    """
    Produces a multipanel plot of an individual sample (image tile), its
    label, its prediction and analytics
    """
    img_size = np.prod(y.shape[:2])
    dim_yscore = yscore.shape[2]
    if dim_yscore > 1:
        type_model = "multiclass"
    else:
        type_model = "binary"
    # retrieve plot properties of different classes
    class_plot_prop = get_class_plot_prop()
    # reshaped versions for metric
    y_reshaped = y.reshape((img_size, 1), order = 'C')
    yscore_reshaped = yscore.reshape((img_size, dim_yscore), order = 'C')
    # versions for display in which we get rid of upper small percentile 
    yscore_plot = np.copy(yscore)
    prc = np.percentile(yscore_plot, [99.9])
    yscore_plot[yscore_plot >= prc] = prc
    # ----------------set up figure ---------------
    fig_sample, axs = plt.subplots(2, 4, figsize=(15, 7))
    fig_sample.suptitle(title, fontsize=16)
    # plot rgb part of image
    show_tile(x[:,:,[2, 1, 0]], axs[0,0], title="RGB");
    # nir
    show_tile(x[:,:,3], axs[0,1], cmap="gray",  title="infrared");
    # labels § to be replaced by Lisa's code
    show_tile(y, axs[0,2], cmap="gray",  title="labels");

    if type_model == "binary":
        cmap = class_plot_prop["any_road"][1]
        # y score (prediction)
        show_tile(yscore_plot, axs[0,3], cmap=cmap, show_colorbar=True,  title="prediction");
        # pale rgb + transparent prediction
        show_tile(exposure.adjust_gamma(x[:,:,[2, 1, 0]], 0.5), axs[1,0]);
        show_tile(yscore_plot, axs[1,0], cmap=cmap, title="rgb + prediction", alpha=.5);
    else:
        cmap = class_plot_prop["paved_road"][1]
        # assume that the first layer represents scores for no_road
        # y score (prediction)
        show_tile(yscore_plot[:,:,1], axs[0,3], cmap=cmap, show_colorbar=True, title="prediction (paved roads)");
        cmap = class_plot_prop["unpaved_road"][1]
        show_tile(yscore_plot[:,:,2], axs[1,0], cmap=cmap, show_colorbar=True, title="prediction (unpaved roads");
        
        
    # auc_roc, auc_pr
    (fpr_sample_dict,
    tpr_sample_dict,
    roc_auc_sample_dict,
    precision_sample_dict,
    recall_sample_dict,
    pr_auc_sample_dict,
    _, _,
    reduced_label_sample_dict) = multiclass_roc_pr(y_reshaped, yscore_reshaped, class_dict=get_class_dict(type_model))
    plot_pr(recall_sample_dict, precision_sample_dict, pr_auc_sample_dict, None, None, axs[1, 2])
    plot_roc(fpr_sample_dict, tpr_sample_dict, roc_auc_sample_dict, axs[1, 3])
    plt.show()
    return fig_sample