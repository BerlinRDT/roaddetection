import logging

import rasterio as rio

from rasterio import windows

import geopandas as gp
import fiona
import numpy as np
from itertools import product
from utils import get_tile_prefix


class Raster(object):

    def __init__(self, imageFile, meta):
        self.logger = logging.getLogger(__name__)

        self.imageFile = imageFile
        self.meta = meta

    def inner_bbox(self, file):
        fiona.drvsupport.supported_drivers['GML'] = 'rw'
        data = gp.read_file(self.meta)
        coordString = data.coordinates.values[0]
        coordinateList = coordString.split(' ')

        innerBoundingBox = []
        for coordinate in coordinateList:
            x = [float(i) for i in coordinate.split(',')]
            innerBoundingBox.append(x)
        return innerBoundingBox

    def get_windows(self, raster, width, height):
        nols, nrows = raster.meta['width'], raster.meta['height']
        offsets = product(range(0, nols, width), range(0, nrows, height))
        big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
        for col_off, row_off in offsets:
            window = windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(
                big_window)
            transform = windows.transform(window, raster.transform)
            yield window, transform

    def to_tiles(self, output_path, window_size):
        logging.info("Generating tiles for image : {}".format(self.imageFile.name))

        i = 0
        with rio.open(self.imageFile) as inds:
            # innerBBox = self.inner_bbox(file)
            meta = inds.meta.copy()
            for window, transform in self.get_windows(inds, window_size, window_size):
                w = inds.read(window=window)
                if not self.is_window_empty(w):
                    meta['transform'] = transform
                    meta['width'], meta['height'] = window.width, window.height
                    self.write_tile(w, meta, output_path, i)
                    i += 1
                    # write_map(inds, window, meta, ax, innerBBox)

    def write_tile(self, window, meta, output_path, i):
        TRAINING_SAT_DIR = '{}/sat'.format(output_path)
        output_tile_filename = '{}/{}_{}.tif'
        outpath = output_tile_filename.format(TRAINING_SAT_DIR, get_tile_prefix(self.imageFile.name), i)
        with rio.open(outpath, 'w', **meta) as outds:
            outds.write(window)

    def is_window_empty(self, w):
        return not np.any(w)
