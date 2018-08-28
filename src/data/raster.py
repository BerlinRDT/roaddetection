import logging
# from shapely.ops import transform
import numpy as np
from shapely.geometry import mapping
import rasterio as rio
from rasterio import windows
from rasterio import features
from rasterio.warp import transform_bounds
from skimage import exposure

from itertools import product
from utils import output_sat_path, output_sat_rgb_path, output_map_path
from bounding_box import inner_bbox, window_trueBoundingBox, cut_linestrings_at_bounds
# import pyproj
# from functools import partial
# from operator import is_not


class Raster(object):

    def __init__(self, analyticFile, rgbFile, meta):
        self.logger = logging.getLogger(__name__)

        self.analyticFile = analyticFile
        self.rgbFile = rgbFile
        self.meta = meta
        self.DST_CRS = "EPSG:4326"

    def get_windows(self, raster, width, height, overlap=None):
        """
        Produces regularly spaced tiles (instances of rasterio.window.Window) 
        of size [width, height] pixels from rasterio DataSetReader (image file opened 
        with rasterio.open) with overlap 
        width: width of tiles (pixels)
        height: height of tiles (pixels)
        overlap: overlap of tiles in each dimension (relative units, range [0 1[)
        """    
        # check overlap
        if overlap is None:
            overlap = 0.0
        else:
            assert (overlap >= 0.0) and (overlap < 1.0), "overlap must be in [0 1["
        
        width_masterImg, height_masterImg = raster.meta['width'], raster.meta['height']
        
        # check that tiles are within master image
        assert (width <= width_masterImg) and (height <= height_masterImg), "tiles are too large for image"
        
        # produce a list of regularly spaced horizontal offsets such that all tiles produced 
        # from it fit into the master image, plus one offset which ensures that the right edge 
        # of the last tile is identical to the right edge of the master window (the implication
        # is that the last tile has a different degree of overlap with its neighbor than the other
        # images)
        offsets_horiz = list(range(0, width_masterImg, round(width *(1.0 - overlap))))
        offsets_horiz[-1] = width_masterImg - width
        # same for vertical offsets
        offsets_vert = list(range(0, height_masterImg, round(height * (1.0 - overlap))))
        offsets_vert[-1] = height_masterImg - height
        # construct iterator
        offsets = product(offsets_horiz, offsets_vert)
        
        for col_off, row_off in offsets:
            window = windows.Window(col_off=col_off, row_off=row_off, width=width, height=height)
            transform = windows.transform(window, raster.transform)
            yield window, transform

    def scale_and_typecast(self, img_arr, meta, mask, dtype, scaling_type):
        """
        Scales values in numpy array img_arr, representing an image obtained via a rasterio 
        read operation, converts their number type and returns the array thus altered.
        img_arr: input array (different color bands are in the first dimension!)
        meta: meta data from underlying file
        dtype: string, number type of output array, e.g. "uint8"
        scaling_type: determines how pixel intensity values are scaled, legal 
        values are 'percentile' and 'equalize_adapthist'
        """
        # scale 
        logging.info("Scaling image using method {}".format(scaling_type))  
        # percentile-based method: band by band
        if scaling_type == "percentile":
            prc = np.zeros([meta["count"]])
            for band_ix in range(0, meta["count"]):
                prc[band_ix] = np.percentile(img_arr[band_ix,:,:], (99.9,))
                img_arr[band_ix] = exposure.rescale_intensity(img_arr[band_ix,:,:], in_range=(0, prc[band_ix]))
            # scale down for type cast
            if dtype == meta["dtype"]:
                pass
            elif ((dtype == 'uint8') and (meta["dtype"] == 'uint16')):
                img_arr = img_arr >> 8
            else:
                raise Exception("scaling for any cast other than uint16->uint8 not yet defined")
        elif scaling_type == "equalize_adapthist":
            # convert to float temporarily
            img_arr = img_arr.astype("float32")
            meta["dtype"] = "float32"
            for band_ix in range(0, meta["count"]):
                img_arr[band_ix] = img_arr[band_ix]/img_arr[band_ix].max()
                img_arr[band_ix] = exposure.equalize_adapthist(img_arr[band_ix], clip_limit=0.03)
            # scaling
            img_arr = (img_arr * np.iinfo(dtype).max)
         # apply mask
        img_arr = img_arr * mask
        if dtype != meta["dtype"]:
            logging.info("Converting from {} to {}".format(meta["dtype"], dtype))
            # type cast 
            img_arr = img_arr.astype(dtype)
            # don't forget to adjust metadata
            meta["dtype"] = dtype
        return img_arr, meta

    def to_tiles(self, output_path, window_size, idx, overlap, dtype, scaling_type):
        logging.info("Generating tiles for image {} with edge length {} and relative edge overlap {}"\
                     .format(self.analyticFile.name, window_size, overlap))        
        i = 0
        with rio.open(self.analyticFile) as raster:
            innerBBox = inner_bbox(self.meta)
            meta = raster.meta.copy()            
            # open and read full image file
            fullImg = raster.read()
            # open and read mask (one mask for whole image, not for bands separately):
            # per definition, it contains 0 for invalid and 255 for valid pixels
            mask = raster.dataset_mask()
            # modify mask such that it contains 1 for valid pixels
            mask[mask > 0] = 1
            # reshape such that it can be multiplied with image 
            mask = mask.reshape((1, raster.height, raster.width))
            # scale and convert
            fullImg, meta = self.scale_and_typecast(fullImg, meta, mask, dtype, scaling_type)
            # loop over windows
            for window, t in self.get_windows(raster, window_size, window_size, overlap):
                # convert windows to numpy array slice indexes
                ix = window.toslices()
                # excise data
                w_img = fullImg[:,ix[0], ix[1]]
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

        linesDf = cut_linestrings_at_bounds(sec_WindowImageBBox, intersecting_road_items)

        m2 = meta.copy()
        m2['count'] = 1
        m2['dtype'] = 'uint8'
        nodata = 0

        with rio.open(output_map_path(self.analyticFile, i, output_path), 'w', **m2) as outds:
            if len(linesDf) > 0:
                g2 = linesDf.to_crs(m2['crs'].data)
                burned = features.rasterize(shapes=[(x.geometry, self.get_pixel_value(x.label)) for i, x in g2.iterrows()],
                                            fill=nodata,
                                            out_shape=(window_size, window_size),
                                            all_touched=True,
                                            transform=meta['transform'])
                outds.write(burned, indexes=1)

    def get_pixel_value(self,label):
        if int(label) == 1:
            return 127
        elif int(label) == 2:
            return 255
        return 0

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
