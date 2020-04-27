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
                lines = []
                count = 0
                isOK = True
                try:
                    for feature in geojson:
                        # accept only entries with at least two coordinates
                        coord = feature["geometry"]["coordinates"]
                        if len(coord) >= 2:
                            # Google Earth-derived labels have a 'name' key in 
                            # which the type of road appears at the end of the 
                            # string after a '_', and possibly also a 'label' 
                            # key, whereas QGIS-derived labels have only a 
                            # 'label' key
                            if "name" in feature['properties'].keys():
                                label = get_road_label(feature['properties']['name'])
                            else:
                                label = feature['properties']['label']
                            line = LineString(feature["geometry"]["coordinates"])
                            payload = {'geometry': line, 'label': label}
                            idx.insert(count, line.bounds, obj=payload)
                            count += 1
                except: 
                    isOK = False
            if not isOK:
                raise Exception("Mishap occurred with file {}".format(file.name))        
    return idx


def get_road_label(prop_name):
    labels = prop_name.split('_')
    return labels[2] if len(labels) > 2 else 2
