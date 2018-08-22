from pathlib import PurePosixPath


def get_meta_data_filename(input_path, rasterFileName):
    """
    Returns path to metadata file. Assumptions about rasterFileName:
       - ends in .tif(f)
    """
    metadata_file = rasterFileName.rsplit('.tif')[0] \
                    + '_metadata.xml'
    return PurePosixPath(input_path, metadata_file)

def get_rgb_filename(input_path, rasterFileName):
    """
    Returns path to rgb visual file.
    """
    rgb_file_name =rasterFileName.rsplit("_", 1)[0].rsplit("_AnalyticMS")[0] \
                    + '_newVisual.tif'
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
    output_tile_filename = '{}/{}_{}.tif'
    outpath = output_tile_filename.format(TRAINING_SAT_DIR, get_tile_prefix(analyticFile.name), i)
    return outpath

def output_sat_rgb_path(analyticFile, i, output_path):
    TRAINING_SAT_RGB_DIR = '{}/sat_rgb'.format(output_path)
    output_tile_filename = '{}/{}_{}.tif'
    outpath = output_tile_filename.format(TRAINING_SAT_RGB_DIR, get_tile_prefix(analyticFile.name), i)
    return outpath

def output_map_path(analyticFile, i, output_path):
    TRAINING_MAP_DIR = '{}/map'.format(output_path)
    output_tile_filename = '{}/{}_{}.tif'
    outpath = output_tile_filename.format(TRAINING_MAP_DIR, get_tile_prefix(analyticFile.name), i)
    return outpath
