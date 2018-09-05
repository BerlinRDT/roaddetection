import logging
from pathlib import Path
import os, shutil
import click
import random

from src.data.utils import get_tile_prefix


@click.command()
@click.argument('raw_images_path', type=click.Path(exists=True))
@click.argument('train_dir', type=click.Path(exists=True))
@click.argument('validation_dir', type=click.Path(exists=True))
@click.argument('test_dir', type=click.Path(exists=True))
def main(raw_images_path, train_dir, validation_dir, test_dir):
    logger = logging.getLogger(__name__)

    logger.info(
        "Moving files from {} to validation dir : {} and test dir : {}".format(train_dir, validation_dir, test_dir))

    file_prefixes = get_analytic_tile_prefixes(raw_images_path)

    random.seed(42)  # should not be changed
    idx = random.sample(range(1, 170), 50)
    val_idx, test_idx = idx[0:25], idx[25:50]

    move_validation_tiles(train_dir, validation_dir, val_idx, file_prefixes)
    move_test_tiles(train_dir, test_dir, test_idx, file_prefixes)

    for directory in [train_dir, validation_dir, test_dir]:
        for file_type in ["sat", "map", "sat_rgb"]:
            target = os.path.join(directory, file_type)
            logger.info("{} : {}".format(target, len(os.listdir(target))))

    logger.info("Done.")


def get_analytic_tile_prefixes(raw_images_path):
    file_prefixes = [get_tile_prefix(r_analytic.name)
                     for r_analytic in Path(raw_images_path).iterdir()
                     if should_make_tiles_from(r_analytic.name)
                     ]
    return file_prefixes


def should_make_tiles_from(r_analytic_name):
    is_analytic_tif = r_analytic_name.endswith(
        ('AnalyticMS.tif', 'AnalyticMS_SR.tif', 'AnalyticMS.tiff', 'AnalyticMS_SR.tiff')
    )
    return is_analytic_tif


def move_test_tiles(train_dir, test_dir, test_idx, file_prefixes):
    test_fnames = ["{0:s}_{1:04d}.tif".format(prefix, i) for i in test_idx for prefix in file_prefixes]
    for fname in test_fnames:
        for file_type in ["sat", "map", "sat_rgb"]:
            src = os.path.join(train_dir, file_type, fname)
            dest = os.path.join(test_dir, file_type, fname)
            if (os.path.exists(src)):
                shutil.move(src, dest)


def move_validation_tiles(train_dir, validation_dir, val_idx, file_prefixes):
    validation_fnames = ["{0:s}_{1:04d}.tif".format(prefix, i) for i in val_idx for prefix in file_prefixes]
    for fname in validation_fnames:
        for file_type in ["sat", "map", "sat_rgb"]:
            src = os.path.join(train_dir, file_type, fname)
            dest = os.path.join(validation_dir, file_type, fname)
            if (os.path.exists(src)):
                shutil.move(src, dest)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
