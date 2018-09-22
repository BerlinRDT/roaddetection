#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This is a collection of functions for the visualization of satellite
# images and quantities derived from them

import numpy as np
import pandas as pd
import rasterio.plot
from skimage import exposure
from sklearn import decomposition
from scipy import stats
import matplotlib.pyplot as plt

def define_band_colors():
    """
    Define dictionary of colors to be used in plots (e.g. histograms) for representing 
    data in specific bands
    """
    band_color_rep = {'B': 'blue',
                      'G': 'green',
                      'R': 'red',
                      'N': 'black',
                      'A': None}
    return band_color_rep


def get_band_properties(src, src_type, band_color_rep):
    """
    Sets up dataframe characterizing data in each band
    src: handle to file opened with rasterio.read()
    src_type: any of "RGB", "RGBA", "BGRN"
    band_color_rep: dict assigning plot colors to bands (e.g. 'R': 'red')
    """
    band_info = pd.DataFrame({"band_index": src.indexes, 
                        "name": list(src_type),
                        "plot_col": [band_color_rep[b] for b in list(src_type)],
                        "dtype": src.dtypes,
                        "maxval": [np.iinfo(t).max for t in src.dtypes],
                        "nodatavals": src.nodatavals,
                       }).set_index("name")
    return band_info


def hist_and_mode(img_arr, band_info, do_visualize=False):
    # visualize histograms & compute mode of each band
    # determine downsampling factor along each axis depending on image size:
    np.prod(img_arr.shape[:2])
    ds_fac = np.max([round(((np.prod(img_arr.shape[:2])) // 1e5) ** 0.5).astype(int), 1])
    md = []
    for band_ix in band_info["band_index"].values:
        ix = band_ix - 1
        sub_data = img_arr[::ds_fac, ::ds_fac, ix].ravel()
        if do_visualize:
            # § better use scikit hist, could be more efficient on integers
            ax = plt.hist(sub_data, 100, histtype='step', color=band_info["plot_col"][ix])
        if "int" in band_info["dtype"][ix]:
            # eliminate zeros for computing mode
            sub_data = sub_data[sub_data>0]
            md.append(stats.mode(sub_data, nan_policy="omit").mode[0])
        else:
            raise Exception("mode not yet impemented for floating point image data")
    if do_visualize:
        plt.xlabel("pixel intensity value")
        plt.ylabel("N (downsampled)")


def convert_sat_img(img_arr, src, src_type, mask_arr=None, pca=False,  \
                    scaling_type=["percentile"], do_scale_bands_separate=True,
                    percentile=99.5, clip_limit=0.03, do_histogram=False):
    """
    Converts analytic satellite image (3 or 4 bands) to visual format, uint8
    INPUT ARGS:
    img_arr: numpy array, resulting from rasterio.read(), that is, bands are
        in the first dimension
    src: pointer to opened file
    src_type: string, bands in source file; any of RGB, RGBA, BGRN 
        (pay attention to order):
        - RGB: normal red green blue image
        - RGBA: RGB with alpha map
        - BGRN: 4-band image with N=near infrared
    mask_arr: numpy array, resulting from rasterio read_mask()
    pca: object of type sklearn.decomposition.pca.PCA or boolean; if the former,
        pixel values will be transformed accordingly; if boolean and True, PCA 
        of pixel values will be computed; additional output args will be returned 
        (see below)
    scaling_type: list or set of types of image scaling to be applied, currently 
        any combination of ("percentile", "equalize_adapthist")
    do_scale_bands_separate: if True, image scaling will be applied to all bands
        separately, potentially resulting in a shift of hue
    percentile: percentile beyond which values will be clipped if scaling_type is
        "percentile"; default is 99.5
    clip_limit: input into exposure.equalize_adapthist
    do_histogram: bool; if True, histograms of pixel intensity values before and
        after scaling will be plotted
    
    RETURNS, in this order:
    img_rgb: rgb image
    img_n: infrared channel (if present in input, otherwise None)
    img_nrg: infrared-red-green array (if infrared is present in input, otherwise none)
    If input arg do_pca is true, additionally, the following outputs will be returned:
    img_pc:  false-color image of first three principal components
    pca: an object of type sklearn.decomposition.pca.PCA that can be used as an input 
        to this function to transform further image data in a reproducible manner
    """
    # input error checking:
    assert(src_type in ("RGB", "RGBA", "BGRN")), "src_type must be either of 'RGB', 'RGBA', 'BGRN'"
    # ensure that actual and expected number of input bands match
    assert(len(src_type)==src.count), "number of bands found in image does not square with src_type"
    assert(not set(scaling_type).difference(("percentile", "equalize_adapthist"))), \
    "scaling_type must be a LIST; legal entries are 'percentile' and 'equalize_adapthist'"
    if mask_arr is None:
        print("image transformations may not yield desired results due to absence of mask_arr")
    if isinstance(pca, decomposition.pca.PCA):
        do_pca = True
    elif isinstance(pca, bool):
        do_pca = pca
    else:
        do_pca = False
    if do_pca:
        assert(src_type=="BGRN"), "PCA not implemented for image types other than 'BGRN'"
    
    # preparations:
    band_color_rep = define_band_colors()
    band_info = get_band_properties(src, src_type, band_color_rep)
    # list of indexes to RGB and N bands, because these bands will be treated
    # differently from other bands
    rgb_ix = [src_type.find(c) for c in "RGB"]
    n_ix = src_type.find("N")
    # set output vars to default None
    img_arr_n = None
    img_arr_nrg = None
    if do_pca:
        img_arr_pc = None
    
    # first thing to do: reshape
    img_arr = rasterio.plot.reshape_as_image(img_arr)
    if mask_arr is not None:
        mask_arr = rasterio.plot.reshape_as_image(mask_arr)
    if do_histogram:
        ax1 = plt.subplot(2,1,1)
        hist_and_mode(img_arr, band_info, do_visualize=True)
        plt.title("before scaling")
    
    # scale
    if do_scale_bands_separate:
        prc = np.zeros([src.count])
        if "percentile" in scaling_type:
            for band_ix in range(src.count):
                prc[band_ix] = np.percentile(img_arr[:,:,band_ix], (percentile,))
                img_arr[:,:,band_ix] = exposure.rescale_intensity(img_arr[:,:,band_ix], in_range=(0, prc[band_ix]))
            # scale to uint8 by doing a bit-wise shift
            img_arr = img_arr >> 8
        if "equalize_adapthist" in scaling_type:
            # convert to float
            img_arr = img_arr.astype('float32')
            for band_ix in range(src.count):
                img_arr[:,:,band_ix] = img_arr[:,:,band_ix]/img_arr[:,:,band_ix].max()
                img_arr[:,:,band_ix] = exposure.equalize_adapthist(img_arr[:,:,band_ix], clip_limit=clip_limit)
            # scale to uint8 range
            img_arr = (img_arr * np.iinfo(np.uint8).max)
        # apply mask
        # NOTE: if img_arr is a numpy masked array, it will remain so after the "percentile"
        # scaling above, but not after the "equalize_adapthist" procedure because scikit-learn
        # does not handle masked arrays, so to make sure that non-image values remain so apply
        # mask here 
        if mask_arr is not None:
            for band_ix in range(src.count):
                img_arr[np.logical_not(mask_arr[:,:,band_ix].astype(np.bool)), band_ix] = band_info["nodatavals"][band_ix]
            
        # split up
        img_arr_rgb = img_arr[:,:,rgb_ix]
        if n_ix >= 0:
            img_arr_n = img_arr[:,:,n_ix]
        
    else:
        # a flag indicating whether image array has been split up into rgb and infrared parts
        flag_band_split = False
        # perform scaling on rgb bands in their entirety, so we don't cause a color shift,
        # but treat infrared band separately because its values are usually far off
        if "percentile" in scaling_type:
            flag_band_split = True
            prc_rgb = np.percentile(img_arr[:,:,rgb_ix], (percentile,))
            # copy rgb bands in rgb order and rescale such that 
            # percentile computed above is the new maximum of the number type present
            img_arr_rgb = exposure.rescale_intensity(img_arr[:,:,rgb_ix], in_range=(0,prc_rgb[0]))
            img_arr_rgb = img_arr_rgb >> 8
            # do the same thing for N band if it exists
            if n_ix >= 0:
                prc_n = np.percentile(img_arr[:,:,n_ix], (99.9,))
                img_arr_n = exposure.rescale_intensity(img_arr[:,:,n_ix], in_range=(0, prc_n[0]))
                img_arr_n = img_arr_n  >> 8        
        if "equalize_adapthist" in scaling_type:
            if flag_band_split:
                img_arr_rgb = img_arr_rgb.astype('float32')/prc_rgb
                if n_ix >= 0:
                    img_arr_n = img_arr_n.astype('float32')/prc_n
            else:
                img_arr = img_arr.astype('float32')
                # normalize and split up
                img_arr_rgb = img_arr[:,:,rgb_ix]/img_arr[:,:,rgb_ix].max()
                if n_ix >= 0:
                    img_arr_n = img_arr[:,:,n_ix]/img_arr[:,:,n_ix].max()
            img_arr_rgb = exposure.equalize_adapthist(img_arr_rgb, clip_limit=clip_limit)
            if n_ix >= 0:
                img_arr_n = exposure.equalize_adapthist(img_arr_n, clip_limit=clip_limit)
            # scale to uint8 range
            img_arr_rgb = (img_arr_rgb * np.iinfo(np.uint8).max)
            if n_ix >= 0:
                img_arr_n = (img_arr_n * np.iinfo(np.uint8).max)            
    
    # apply mask (see note above, same here)
    if mask_arr is not None:
        for i, band_ix in enumerate(rgb_ix):
            # note that bands have been 'sorted' in img_arr_rgb, but not mask_arr,
            # hence the different indices
            img_arr_rgb[np.logical_not(mask_arr[:,:,band_ix].astype(np.bool)), i] = band_info["nodatavals"][band_ix]
        if n_ix >= 0:
            img_arr_n[np.logical_not(mask_arr[:,:,n_ix].astype(np.bool))] = band_info["nodatavals"][n_ix]

    if do_histogram:
        if n_ix >= 0:
            # cat all bands
            img_arr_plot = np.concatenate((img_arr_rgb, img_arr_n.reshape(img_arr_n.shape + (1,))), axis=2)
            band_info_plot = band_info.iloc[rgb_ix + [n_ix],:]
        else:
            img_arr_plot = img_arr_rgb
            band_info_plot = band_info.iloc[rgb_ix,:]
        ax2 = plt.subplot(2,1,2)
        hist_and_mode(img_arr_plot, band_info_plot, do_visualize=True)
        plt.title("after scaling")
        
    # compute PCA
    # Note: standardization is omitted deliberately because following the procedures
    # above the values in the different bands should be roughly in the same ranges
    if do_pca:
        # only if pca is True create new pca instance, otherwise use the one provided as input arg
        if pca:
            pca = decomposition.PCA(n_components=3)
        # cat all bands
        img_arr_pc = np.concatenate((img_arr_n.reshape(img_arr_n.shape + (1,)), img_arr_rgb), axis=2)
        # reshape such that pixel intensity values are the features
        img_arr_pc = img_arr_pc.reshape((np.prod(img_arr_pc.shape[:2]), 4), order='C')
        # reshape mask in a similar manner (note that we don't care about band 
        # order in the mask here because pixels with any foul value are kicked out)
        if mask_arr is not None:
            good_ix = mask_arr.astype(np.bool)
            good_ix = good_ix.reshape((np.prod(mask_arr.shape[:2]), mask_arr.shape[2]), order='C')
            good_ix = np.all(good_ix, axis=1)
            pca.fit(img_arr_pc[good_ix,:])
        else:
            pca.fit(img_arr_pc)
        # generate scores
        img_arr_pc = pca.transform(img_arr_pc)
        # apply mask (again)
        if mask_arr is not None:
            img_arr_pc[np.logical_not(good_ix), :] = 0.0

        # shift to positive values, scale
        img_arr_pc = img_arr_pc - img_arr_pc.min()
        img_arr_pc = img_arr_pc / img_arr_pc.max()
        img_arr_pc = (img_arr_pc * np.iinfo(np.uint8).max)
        
        print("explained variance: {}".format(pca.explained_variance_ratio_))
        # reshape back, keeping only three layers
        img_arr_pc = img_arr_pc.reshape(img_arr_rgb.shape[:2] + (3,), order='C')

    # conversion to uint8
    img_arr_rgb = img_arr_rgb.astype("uint8")
    if n_ix >= 0:
        img_arr_n = img_arr_n.astype("uint8")
    if do_pca:
        img_arr_pc = img_arr_pc.astype("uint8")
    
    # create false color image
    if n_ix >= 0:
        img_arr_nrg = np.concatenate((img_arr_n.reshape(img_arr_n.shape + (1,)), img_arr_rgb[:,:,:2]), axis=2)
        
    if do_pca:
        return img_arr_rgb, img_arr_n, img_arr_nrg, img_arr_pc, pca
    else:
        return img_arr_rgb, img_arr_n, img_arr_nrg


