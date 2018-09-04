import logging
from pathlib import Path
import skimage.io as io
import os, shutil
import numpy as np
import click
import random


@click.command()
@click.argument('tiles_folder', type=click.Path(exists=True))
@click.option('--spare', default=5, help='Precentage of empty tiles to spare')
def main(tiles_folder, spare):
    """
    :param tiles_folder: Tiles Directory containing (sat, map, sat_rgb) to remove empty tiles from
    :param spare: Percentage of empty tiles to spare
    :return:
    """
    logger = logging.getLogger(__name__)
    logger.info("Deleting empty tiles from folder : {} sparing {} %".format(tiles_folder, spare))

    tiles_with_no_roads = [
        fname.name
        for fname in Path(os.path.join(tiles_folder, 'map')).iterdir()
        if should_delete(fname)
    ]
    number_of_no_road_tiles = len(tiles_with_no_roads)
    spared = random.sample(tiles_with_no_roads, int(number_of_no_road_tiles * spare / 100))

    tiles_with_no_roads = set(tiles_with_no_roads)
    spared = set(spared)
    to_be_deleted = tiles_with_no_roads - spared

    logger.info("In directory {} Number of empty tiles : {} Deleting : {} sparing : {}".format(
        tiles_folder, number_of_no_road_tiles, len(to_be_deleted), len(spared)))

    if click.confirm('This action cannot be undone'.format(tiles_folder)):
        for fname in to_be_deleted:
            for file_type in ["sat", "map", "sat_rgb"]:
                src = os.path.join(tiles_folder, file_type, fname)
                if (os.path.exists(src)):
                    os.remove(src)

        click.echo("Deleted {} tiles with no roads".format(len(to_be_deleted)))


def should_delete(fname):
    return not np.any(io.imread(fname.as_posix()))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
