import logging
from shapely.ops import transform
from shapely.geometry import mapping

import rasterio as rio

from rasterio import windows
from rasterio import features
from rasterio.warp import transform_bounds

import numpy as np
from itertools import product
from utils import output_sat_path, output_sat_rgb_path, output_map_path
from bounding_box import inner_bbox, window_trueBoundingBox, cut_linestrings_at_bounds
import pyproj
from functools import partial
from operator import is_not


class Raster(object):

    def __init__(self, analyticFile, rgbFile, meta):
        self.logger = logging.getLogger(__name__)

        self.analyticFile = analyticFile
        self.rgbFile = rgbFile
        self.meta = meta
        self.DST_CRS = "EPSG:4326"

    def get_windows(self, raster, width, height):
        nols, nrows = raster.meta['width'], raster.meta['height']
        offsets = product(range(0, nols, width), range(0, nrows, height))
        big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
        for col_off, row_off in offsets:
            window = windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(
                big_window)
            transform = windows.transform(window, raster.transform)
            yield window, transform

    def to_tiles(self, output_path, window_size, idx):
        logging.info("Generating tiles for image : {}".format(self.analyticFile.name))

        i = 0
        with rio.open(self.analyticFile) as raster:
            innerBBox = inner_bbox(self.meta)
            meta = raster.meta.copy()
            for window, t in self.get_windows(raster, window_size, window_size):
                w_img = raster.read(window=window)
                if not self.is_window_empty(w_img):
                    meta['transform'] = t
                    meta['width'], meta['height'] = window.width, window.height
                    self.write_analytic_tile(w_img, meta, output_path, i)
                    self.write_rgb_tile(window, meta, output_path, i)
                    self.write_map(raster, window, output_path, idx, i, meta, innerBBox, window_size)
                    i += 1

    def write_map(self, raster, window, output_path, spatial_idx, i, meta, box, window_size):
        windowBounds = self.transform_bnds(raster.crs, self.DST_CRS, raster.window_bounds(window))

        sec_WindowImageBBox = window_trueBoundingBox(windowBounds, box)

        dst_bounds = mapping(sec_WindowImageBBox.geometry)['bbox']

        intersecting_road_items = spatial_idx.intersection(dst_bounds, objects=True)

        lines = [cut_linestrings_at_bounds(sec_WindowImageBBox.geometry.values[0], r.object)
                 for r in intersecting_road_items]
        lines = list(filter(partial(is_not, None), lines))

        m2 = meta.copy()
        m2['count'] = 1
        m2['dtype'] = 'uint8'
        nodata = 255

        with rio.open(output_map_path(self.analyticFile, i, output_path), 'w', **m2) as outds:
            if len(lines) > 0:
                g2 = [transform(self.project(), line) for line in lines]
                burned = features.rasterize(g2,
                                            fill=nodata,
                                            out_shape=(window_size, window_size),
                                            all_touched=True,
                                            transform=meta['transform'])
                outds.write(burned, indexes=1)

    def project(self):
        p1 = pyproj.Proj(init=self.DST_CRS)
        p2 = pyproj.Proj(init='EPSG:32750')  # the is the crs of the source raster file
        project = partial(pyproj.transform, p1, p2)
        return project

    def write_analytic_tile(self, window, meta, output_path, i):
        outpath = output_sat_path(self.analyticFile, i, output_path)
        with rio.open(outpath, 'w', **meta) as outds:
            outds.write(window)

    def write_rgb_tile(self, window, meta, output_path, i):
        outpath = output_sat_rgb_path(self.analyticFile, i, output_path)
        m2 = meta.copy()
        m2['dtype'] = 'uint8'
        with rio.open(self.rgbFile.as_posix()) as raster_rgb:
            w = raster_rgb.read(window=window)
            with rio.open(outpath, 'w', **m2) as outds:
                outds.write(w)

    def is_window_empty(self, w):
        return not np.any(w)

    def transform_bnds(self, src_crs, dst_crs, src_bounds):
        return transform_bounds(src_crs, dst_crs, src_bounds[0], src_bounds[1], src_bounds[2], src_bounds[3])
