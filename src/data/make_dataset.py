# -*- coding: utf-8 -*-

import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from raster import Raster
from utils import get_meta_data
from spatial_index import create_spatial_index
import kml2geojson as k2g


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """
        Runs data processing scripts to turn raw data
         from ({Root}/data/raw) into cleaned data ready to
         be analyzed (saved in {Root}/data/train).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    images_path = "{}/images".format(input_filepath)
    labels_path = "{}/labels".format(input_filepath)

    convert_kml_to_geojson(labels_path)
    idx = create_spatial_index(labels_path)
    make_tiles(images_path, output_filepath, idx)


def make_tiles(images_path, output_filepath, idx):
    for file in Path(images_path).iterdir():
        if file.name.endswith(('.tif', '.tiff')):
            meta_data = get_meta_data(images_path, file.name)
            raster = Raster(file, meta_data)
            raster.to_tiles(output_path=output_filepath, window_size=1024, idx=idx)


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
