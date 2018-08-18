import logging

from pathlib import Path
from shapely.geometry import mapping, LineString, shape, Polygon, box
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
                    LineString(feature["geometry"]["coordinates"])
                    for feature in geojson
                    if len(feature["geometry"]["coordinates"]) >= 2
                ]
                count = -1
                for line in lines:
                    count += 1
                    idx.insert(count, line.bounds, obj=line)
    return idx
