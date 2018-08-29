# -*- coding: utf-8 -*-

import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from raster import Raster
from utils import get_meta_data_filename, get_rgb_filename
from spatial_index import create_spatial_index
import kml2geojson as k2g


@click.command()
@click.option('--window_size', default=512, help='Length of edges of image tiles')
@click.option('--overlap', default=0.25, help='Overlap of edges of image tiles [0.0  1.0[')
@click.option('--scaling_type', default='equalize_adapthist', help='Image scaling: [equalize_adapthist] | percentile')
@click.option('--raw_prefix', default=None, help='Filter (prefix) raw images to be picked up for creating tiles.')
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(window_size, overlap, scaling_type, raw_prefix, input_filepath, output_filepath):
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

	logger.info('making final data set from raw data with filter {}'.format(raw_prefix))
	images_path = "{}/images".format(input_filepath)
	labels_path = "{}/labels".format(input_filepath)

	convert_kml_to_geojson(labels_path)
	idx = create_spatial_index(labels_path)
	make_tiles(images_path, output_filepath, window_size=window_size, idx=idx,
	           overlap=overlap, dtype=dtype, scaling_type=scaling_type, raw_prefix_filter=raw_prefix)


def make_tiles(images_path, output_filepath, window_size, idx, overlap, dtype, scaling_type, raw_prefix_filter):
	for r_analytic in Path(images_path).iterdir():
		if should_make_tiles_from(r_analytic.name, raw_prefix_filter):
			meta_data_filename = get_meta_data_filename(images_path, r_analytic.name)
			r_visual_rgb_filename = get_rgb_filename(images_path, r_analytic.name)
			raster = Raster(r_analytic, r_visual_rgb_filename, meta_data_filename)
			raster.to_tiles(output_path=output_filepath, window_size=window_size, \
			                idx=idx, overlap=overlap, dtype=dtype, scaling_type=scaling_type)


def should_make_tiles_from(r_analytic_name, raw_prefix_filter):
	is_analytic_tif = r_analytic_name.endswith(
		('AnalyticMS.tif', 'AnalyticMS_SR.tif', 'AnalyticMS.tiff', 'AnalyticMS_SR.tiff')
	)
	name_begins_with = r_analytic_name.startswith(raw_prefix_filter) if raw_prefix_filter else True
	return is_analytic_tif & name_begins_with


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
