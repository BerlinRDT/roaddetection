from pathlib import PurePosixPath


def get_meta_data(input_path, rasterFileName):
    metadata_file = rasterFileName.rsplit('newVisual.tif')[0] \
                    + 'AnalyticMS_metadata.xml'
    return PurePosixPath(input_path, metadata_file)


def get_tile_prefix(rasterFileName):
    return rasterFileName.rsplit('newVisual.tif')[0]
