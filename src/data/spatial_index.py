import logging

from pathlib import Path
from shapely.geometry import LineString
from rtree import index
import fiona


def create_spatial_index(labels_path):
    logger = logging.getLogger(__name__)
    logger.info('Creating spatial index from geojsons in {}'.format(labels_path))

    idx = index.Index()
    for file in Path(labels_path).iterdir():
        if file.name.endswith('.geojson'):
            with fiona.open('{}/{}'.format(labels_path, file.name), "r") as geojson:
                lines = [
                    (LineString(feature["geometry"]["coordinates"]), feature['properties']['name'])
                    for feature in geojson
                    if len(feature["geometry"]["coordinates"]) >= 2
                ]
                count = -1
                for line, prop_name in lines:
                    count += 1
                    payload = {'geometry': line, 'label': get_road_label(prop_name)}
                    idx.insert(count, line.bounds, obj=payload)
    return idx


def get_road_label(prop_name):
    labels = prop_name.split('_')
    return labels[2] if len(labels) > 2 else 2
