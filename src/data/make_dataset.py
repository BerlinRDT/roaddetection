# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from raster import Raster
from utils import get_meta_data


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
    for file in Path(input_filepath).iterdir():
        if file.name.endswith(('.tif', '.tiff')):
            meta_data = get_meta_data(input_filepath, file.name)
            raster = Raster(file, meta_data)
            raster.to_tiles(output_path=output_filepath, window_size=1024)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
