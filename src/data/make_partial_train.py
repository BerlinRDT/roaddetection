import logging
from pathlib import Path
import skimage.io as io
import os, shutil
import numpy as np
import click


@click.command()
@click.argument('train_dir', type=click.Path(exists=True))
@click.argument('partial_dir', type=click.Path(exists=True))
@click.option('--threshold', default=2.0, help='Threshold (% of road) to create partial train set from')
@click.option('--window_size', default=512, help='Window size of the tiles in pixels.')
def main(train_dir, partial_dir, threshold, window_size):
    logger = logging.getLogger(__name__)
    logger.info(
        "Creating partial train set from {} in to directory {} with threshold {} window size {}".format(train_dir,
                                                                                                        partial_dir,
                                                                                                        threshold,
                                                                                                        window_size))
    tiles_with_roads_above_threshold = [
        fname.name
        for fname in Path(os.path.join(train_dir, 'map')).iterdir()
        if should_move(fname, threshold, window_size)
    ]

    # if click.confirm('This action cannot be undone'.format(tiles_folder)):
    for fname in tiles_with_roads_above_threshold:
        for file_type in ["sat", "map", "sat_rgb"]:
            src = os.path.join(train_dir, file_type, fname)
            dest = os.path.join(partial_dir, file_type, fname)
            if (os.path.exists(src)):
                logger.debug("moving {} to {}".format(src, dest))
                shutil.copy(src, dest)

    click.echo("Created partial train set with {} tiles".format(len(tiles_with_roads_above_threshold)))


def should_move(fname, threshold, window_size):
    map = io.imread(fname.as_posix())
    above_threshold = (len(map[map != 0.0]) * 100) / (window_size * window_size) >= threshold
    return above_threshold


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
