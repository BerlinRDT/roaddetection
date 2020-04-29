import logging

from pathlib import Path
from shapely.geometry import LineString
from rtree import index
import fiona

# definitions of unpaved and paved roads in open street maps' "fclass" property
OS_UNPAVED = ['service', 'residential', 'track', 'unclassified', 'living_street']
OS_PAVED = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'primary_link', 'secondary_link', 'tertiary_link']
# integer code for paved and unpaved roads
PAVED = "1"
UNPAVED = "2"
ROAD_SET = [PAVED, UNPAVED]

def create_spatial_index(labels_path):
    logger = logging.getLogger(__name__)
    logger.info('Creating spatial index from geojsons in {}'.format(labels_path))
    idx = index.Index()
    for file in Path(labels_path).iterdir():
        if file.name.endswith('.geojson'):
            with fiona.open('{}/{}'.format(labels_path, file.name), "r") as geojson:
                lines = []
                count = 0
                type_count = {PAVED: 0, UNPAVED: 0}
                isOK = True
                try:
                    for feature in geojson:
                        # accept only entries with at least two coordinates
                        coord = feature["geometry"]["coordinates"]
                        if len(coord) >= 2:
                            # retrieve road label
                            label =  get_road_label(feature['properties'])
                            type_count[str(label)] += 1
                            line = LineString(feature["geometry"]["coordinates"])
                            payload = {'geometry': line, 'label': label}
                            idx.insert(count, line.bounds, obj=payload)
                            count += 1
                    logger.info("{}: {} unpaved roads, {} paved roads".\
                                format(file.name, type_count[UNPAVED], type_count[PAVED]))
                except: 
                    isOK = False
            if not isOK:
                raise Exception("Mishap occurred with file {}".format(file.name))        
    return idx


def get_road_label(properties):
    """
    Retrieve road label-label (sic, and sorry) from geojson feature.
    
    The function retrieves the "label" of a single geojson feature (= a 
    multiline marking a road, aka "road label"). 
    The label is an int coding for the type of road (paved, unpaved). Discri-
    mination between the types of road is both a legacy of an earlier stage of 
    the project and a potential investment in the future (in case a classi-
    fication of the roads shall be re-implemented).
    The label is retrieved from one of the following keys:
        - "fclass" (as available for road labels from OSM)
        - "label" (as should be present in own, QGIS-labeled data sets)
        - if none of the above, from "name" (as available in first-generation,
          Google Earth-generated labels)
    
    Input
    -----
        properties: collections.OrderedDict, value of a geojson.feature's 
            "property" key 
    Output
    -----
        int, road label, either PAVED or UNPAVED
    """
    label = []
    # OSM labels should have an fclass property
    if ("fclass" in properties.keys()) and (properties["fclass"] is not None):
        if properties["fclass"] in OS_PAVED:
            label.append(PAVED)
        else:
            label.append(UNPAVED)
    # labels produced manually with QGIS should have a "label"
    if ("label" in properties.keys()) and (properties["label"] is not None):
        label.append(properties['label'])
    # if we have neither fclass nor label, this is an old-style kind of label
    # generated with Google Earth, in which the road label is in the name,
    # namely, after the last '_'
    if len(label) == 0:
        if ("name" in properties.keys()) and (properties["name"] is not None):
            splinters = properties["name"].split("_")
            # if the name does contain at least one underscore, extract the 
            # partial string after its last occurrence sans leading or trailing
            # whitespace
            if len(splinters) >= 2:
                label = splinters[-1].strip()
            else: 
                # non-conformant name: assume unpaved road
                label = UNPAVED
        else:
            # none of the expected keys present or with values: assume unpaved
            label = UNPAVED
    elif len(label) == 1:
        # unambiguous case
        label = label[0]
    elif len(label) == 2:
        # both "fclass" and "label" with values - for now, adhere to "fclass"
        # (we may want to issue a warning in the future)
        label = label[0]
    else:
        raise Exception("check code, length of label outside [0, 1, 2]")
        
    # final check
    if not (label in ROAD_SET):
        # non-conformant entries for any of the keys: assume unpaved road
        label = UNPAVED
    
    return int(label)