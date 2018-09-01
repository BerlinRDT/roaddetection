# -*- coding: utf-8 -*-

import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from raster import Raster
from utils import get_meta_data_filename, get_rgb_filename
from spatial_index import create_spatial_index
import kml2geojson as k2g
import pandas as pd


@click.command()
@click.option('--window_size', default=512, help='Length of edges of image tiles')
@click.option('--overlap', default=0.25, help='Overlap of edges of image tiles [0.0  1.0[')
@click.option('--scaling_type', default='equalize_adapthist', help='Image scaling: [equalize_adapthist] | percentile')
@click.option('--raw_prefix', default=None, help='Filter (prefix) raw images to be picked up for creating tiles.')
@click.option('--region', '-r', default='all', type=click.Choice(['all', 'borneo', 'harz']),
              help='Create tiles from a given region')
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(window_size, overlap, scaling_type, raw_prefix, input_filepath, output_filepath, region):
    """
        Runs data processing scripts to turn raw data
         from ({Root}/data/raw) into cleaned data ready to
         be analyzed (saved in {Root}/data/train).
    """
    logger = logging.getLogger(__name__)

    # number type of output image files
    dtype = "uint8"

    # some error checks
    assert (0.0 <= overlap < 1.0), "overlap must be in [0.0 1.0["
    assert (dtype in ("uint8", "uint16")), "dtype must be 'uint8' or 'uint16'"
    assert (scaling_type in (
        "percentile", "equalize_adapthist")), "scaling_type must be 'percentile' or 'equalize_adapthist'"

    logger.info('making tiles from region {} into folder {} with raw image filter {}'.format(region,
                                                                                             output_filepath,
                                                                                             raw_prefix or "None"))
    images_path = "{}/images".format(input_filepath)
    labels_path = "{}/labels".format(input_filepath)

    convert_kml_to_geojson(labels_path)
    idx = create_spatial_index(labels_path)
    make_tiles(images_path, output_filepath, window_size=window_size, idx=idx,
               overlap=overlap, dtype=dtype, scaling_type=scaling_type,
               raw_prefix_filter=raw_prefix, region_filter=region)


def make_tiles(images_path, output_filepath, window_size, idx, overlap, dtype, scaling_type,
               raw_prefix_filter, region_filter):
    img_list = pd.read_json("list_satellite_images_training.json")

    for r_analytic in Path(images_path).iterdir():
        if should_make_tiles_from(r_analytic.name, raw_prefix_filter, region_filter, img_list):
            meta_data_filename = get_meta_data_filename(images_path, r_analytic.name)
            r_visual_rgb_filename = get_rgb_filename(images_path, r_analytic.name)
            raster = Raster(r_analytic, r_visual_rgb_filename, meta_data_filename)
            raster.to_tiles(output_path=output_filepath, window_size=window_size, idx=idx, overlap=overlap,
                            dtype=dtype,
                            scaling_type=scaling_type)


def should_make_tiles_from(r_analytic_name, raw_prefix_filter, region_filter, img_list):
    return is_analytic_tif(r_analytic_name) and \
           name_begins_with_prefix(r_analytic_name, raw_prefix_filter) and \
           is_raster_from_desired_region(r_analytic_name, region_filter, img_list)


def name_begins_with_prefix(r_analytic_name, raw_prefix_filter):
    return r_analytic_name.startswith(raw_prefix_filter) if raw_prefix_filter else True


def is_analytic_tif(r_analytic_name):
    return r_analytic_name.endswith(
        ('AnalyticMS.tif', 'AnalyticMS_SR.tif', 'AnalyticMS.tiff', 'AnalyticMS_SR.tiff')
    )


def is_raster_from_desired_region(r_analytic_name, region_filter, img_list):
    if (region_filter == "all"):
        return True

    logger = logging.getLogger(__name__)

    filtered_by_region = img_list.loc[img_list.directory.str.contains(region_filter, case=False)]
    shouldIgnore = filtered_by_region.loc[filtered_by_region.analyticImgName.str.contains(r_analytic_name)].empty

    if shouldIgnore:
        logger.info("Ignoring raster {} because of region filter {}".format(r_analytic_name, region_filter))

    return not shouldIgnore


def convert_kml_to_geojson(labels_path):
    logger = logging.getLogger(__name__)

    for file in Path(labels_path).iterdir():
        if file.name.endswith(('.kml', 'kmz')):
            logger.info('Generating geojson from {}'.format(file.name))
            kmlpath = '{}/{}'.format(labels_path, file.name)
            k2g.convert(kmlpath, labels_path)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
