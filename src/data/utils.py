import numpy as np
from pathlib import PurePosixPath
import os

def get_meta_data_filename(input_path, rasterFileName):
    """
    Returns path to metadata file. Assumptions about rasterFileName:
       - ends in .tif(f)
    """
    metadata_file = rasterFileName.rsplit('.tif')[0] + '_metadata.xml'
    return PurePosixPath(input_path, metadata_file)


def get_rgb_filename(input_path, rasterFileName):
    """
    Returns path to rgb visual file.
    """
    rgb_file_name = rasterFileName.rsplit("_", 1)[0].rsplit("_AnalyticMS")[0] + '_newVisual.tif'
    return PurePosixPath(input_path, rgb_file_name)


def get_tile_prefix(rasterFileName):
    """
    Returns 'rump' of raster file name, to be used as prefix for tile files.
    rasterFileName is <date>_<time>_<sat. ID>_<product type>_<asset type>.tif(f)
    where asset type can be any of ["AnalyticMS","AnalyticMS_SR","Visual","newVisual"]
    The rump is defined as <date>_<time>_<sat. ID>_<product type>
    """
    return rasterFileName.rsplit("_", 1)[0].rsplit("_AnalyticMS")[0]

def output_sat_path(analyticFile, i, output_path):
    TRAINING_SAT_DIR = '{}/sat'.format(output_path)
    output_tile_filename = '{0:s}/{1:s}_{2:04d}.tif'
    outpath = output_tile_filename.format(TRAINING_SAT_DIR, get_tile_prefix(analyticFile.name), i)
    return outpath


def output_sat_rgb_path(analyticFile, i, output_path):
    TRAINING_SAT_RGB_DIR = '{}/sat_rgb'.format(output_path)
    output_tile_filename = '{0:s}/{1:s}_{2:04d}.tif'
    outpath = output_tile_filename.format(TRAINING_SAT_RGB_DIR, get_tile_prefix(analyticFile.name), i)
    return outpath


def output_map_path(analyticFile, i, output_path):
    TRAINING_MAP_DIR = '{}/map'.format(output_path)
    output_tile_filename = '{0:s}/{1:s}_{2:04d}.tif'
    outpath = output_tile_filename.format(TRAINING_MAP_DIR, get_tile_prefix(analyticFile.name), i)
    return outpath


def get_list_samplefiles(dir_samples):
    """
    Returns a list of and the number of sample files (satellite image tiles) in given directory
    """
    # list of satellite image files & their number
    _, _, file_list_x = next(os.walk(dir_samples))
    num_x = len(file_list_x)
    return file_list_x, num_x

def gen_sample_index(num_x_available, num_x_use, mode_sample_choice="random", metric=None):
    """
    Returns an array of indexes to samples, given the number of samples available (num_x_available)
    and the number to be used (num_x_use). Indexes can either be random or correspond to the head and tail
    of the samples according to a quantitiy listed in metric.
    """
    if mode_sample_choice == "random":
        samples_ix = np.random.choice(num_x_available, num_x_use, replace=False)
    elif mode_sample_choice == "head_tail":
        assert(metric is not None), "'head_tail' mode of choosing samples requires a metric"
        # indexes to best and worst examples
        ix_sorted = np.argsort(metric)
        samples_ix = np.hstack((ix_sorted[:(num_x_use//2)],ix_sorted[(-num_x_use//2):]))
    else:
        raise Exception("illegal choice for mode_sample_choice")
    return samples_ix