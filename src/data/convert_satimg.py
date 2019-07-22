#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 20:28:01 2019
@author: hh

Code for converting 4-band satellite images (tif) into rgb versions.
Code is adapted from (largely identical) jupyter notebooks named
quick_and_dirty_geotiff_viewer_LISA.ipynb, image_conversion_tovisual_hh.ipynb,
and transform_Analytic2Visible.ipynb (in notebooks/visualize_imagery).
Plotting and histogram code has been omitted.
"""

from pathlib import Path
import logging
import numpy as np
import rasterio as rio
from skimage import exposure
import click

def transform2visible(inds, bands):
    dictBands = {'b': (0,2), 'g': (1,1), 'r': (2,0)}
    imgArr = inds.read(masked=True)
    msk = inds.read_masks()
    sz = np.shape(imgArr)
    visArray = np.zeros((sz[1], sz[2], 4), 'uint8')

    visArray[..., 3] = msk[0]

    for k in bands:
        planetScopeIndex, RGBIndex  = dictBands[k]
        # Normalize each band-image
        ima = (imgArr[planetScopeIndex][:,:])
        img = (ima.astype('float')/ima.max())

        # Adaptive Equalization
        img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
        masked_img_adapteq = np.ma.masked_array(img_adapteq, np.logical_not(msk[0]))
        np.ma.set_fill_value(masked_img_adapteq, 0)
        visArray[..., RGBIndex] = masked_img_adapteq.filled()*255
    return visArray


def read_transform_img(file):
    with rio.open(file) as inds:
        meta = inds.meta.copy()
        meta['dtype'] = 'uint8'
        rgb_img = transform2visible(inds, ['b', 'g', 'r'])
    return rgb_img, meta


def write_rgb_img(file, in_path, rgb_img, meta):
    file_name_rgb = in_path + '/' + file.name.rsplit('AnalyticMS.tif')[0]+'newVisual.tif'
    with rio.open(file_name_rgb, 'w', **meta) as outds:
        sz = np.shape(rgb_img)
        output  = np.zeros((4, sz[0], sz[1]), 'uint8')
        for i in range(4):
            output[i] = rgb_img[..., i]
        outds.write(output.astype(np.uint8))


@click.command()
@click.argument('in_path', type=click.Path(exists=True))
def main(in_path):
    logger = logging.getLogger(__name__)
    for file in Path(in_path).iterdir():
        if file.name.endswith(('AnalyticMS.tif', 'AnalyticMS.tiff')):
            logger.info('converting {}...'.format(file.name))
            rgb_img, meta = read_transform_img(file)
            write_rgb_img(file, in_path, rgb_img, meta)


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
