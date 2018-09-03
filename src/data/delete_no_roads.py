import logging
from pathlib import Path
import skimage.io as io
import os, shutil
import numpy as np
import click


@click.command()
@click.argument('tiles_folder', type=click.Path(exists=True))
def main(tiles_folder):
    logger = logging.getLogger(__name__)
    logger.info("Deleting empty tiles from folder : {}".format(tiles_folder))
    tiles_with_no_roads = [
        fname.name
        for fname in Path(os.path.join(tiles_folder, 'map')).iterdir()
        if should_delete(fname)
    ]

    if click.confirm('This action cannot be undone'.format(tiles_folder)):
        for fname in tiles_with_no_roads:
            for file_type in ["sat", "map", "sat_rgb"]:
                src = os.path.join(tiles_folder, file_type, fname)
                if (os.path.exists(src)):
                    os.remove(src)

        click.echo("Deleted {} tiles with no roads".format(len(tiles_with_no_roads)))


def should_delete(fname):
    return not np.any(io.imread(fname.as_posix()))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
