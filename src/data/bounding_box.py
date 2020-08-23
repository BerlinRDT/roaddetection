from shapely.geometry import mapping, shape, Polygon, box, GeometryCollection
from shapely import wkt

import geopandas as gp
import fiona

# Note: the code line marked OLD below creates this warning:    
# /home/hh/miniconda3/envs/roaddetection/lib/python3.7/site-packages/pyproj/crs/crs.py:53: FutureWarning: '+init=<authority>:<code>' syntax is deprecated. '<authority>:<code>' is the preferred initialization method. When making the change, be mindful of axis order changes: https://pyproj4.github.io/pyproj/stable/gotchas.html#axis-order-changes-in-proj-6
# return _prepare_from_string(" ".join(pjargs))        
# The following solution has been found and implemented and seems to work:
# https://stackoverflow.com/a/59703971/10300133    
# OLD:
# gdf_TrueBounds.crs = ({'init': 'epsg:4326'})
# NEW:
# gdf_TrueBounds.crs = "epsg:4326"

def inner_bbox(metadata):
    fiona.drvsupport.supported_drivers['GML'] = 'rw'
    data = gp.read_file(metadata.as_posix())
    coordString = data.coordinates.values[0]
    coordinateList = coordString.split(' ')

    innerBoundingBox = []
    for coordinate in coordinateList:
        x = [float(i) for i in coordinate.split(',')]
        innerBoundingBox.append(x)
    return innerBoundingBox


def bounds2box(bounds):
    xmin, ymin, xmax, ymax = bounds
    bound = gp.GeoDataFrame(geometry=[box(xmin, ymin, xmax, ymax)])
    bound.crs = "epsg:4326"
    return bound


def bounds2polygon(metafile_data):
    bounds = gp.GeoDataFrame(geometry=[Polygon(metafile_data)])
    bounds.crs = "epsg:4326"
    return bounds


# def window_trueBoundingBox(windowBox, imageBox):
#    gdf_WindowBounds = bounds2box(windowBox)
#    gdf_ImageBounds = bounds2polygon(imageBox)
#
#    gdf_TrueBounds = gp.GeoDataFrame(gp.overlay(gdf_WindowBounds, gdf_ImageBounds, how='intersection').geometry)
#    gdf_TrueBounds.crs = ({'init': 'epsg:4326'})
#    return gdf_TrueBounds

def window_trueBoundingBox(windowBox, imageBox):
    gdf_WindowBounds = bounds2box(windowBox)
    gdf_ImageBounds = bounds2polygon(imageBox)
    temp = gp.overlay(gdf_WindowBounds, gdf_ImageBounds, how='intersection')
    if len(temp.index) == 0:
        gdf_TrueBounds = gdf_WindowBounds
    else:
        gdf_TrueBounds = gp.GeoDataFrame(gp.overlay(gdf_WindowBounds, gdf_ImageBounds, how='intersection').geometry)
    gdf_TrueBounds.crs = "epsg:4326"
    
    return gdf_TrueBounds


def cut_linestrings_at_bounds(bounds, intersecting_road_items):
    geom = []
    iden = []
    for geo_object in intersecting_road_items:
        road = geo_object.object['geometry']
        label = geo_object.object['label']
        for boundingBox in bounds.geometry.values:
            lineobj = wkt.loads(str(shape(road)))
            conus_transformed_poly = wkt.loads(str(boundingBox))
            conus_intersection = conus_transformed_poly.intersection(lineobj)
            cut_line = shape(mapping(conus_intersection))
            if not cut_line.is_empty and cut_line.is_valid:
                geom.append(buffered_line(cut_line, label))
                iden.append(label)

    gpd = gp.GeoDataFrame({'label': iden}, geometry=geom, crs='epsg:4326')
    # For testing if different colors appear in the rasterized labels
    # if len(gpd) > 3:
    #     gpd.loc['label'] = 1
    return gpd


def buffered_line(cut_line, label):
    buffer_distance = 0.0001
    buffer_distance = buffer_distance / 3 if int(label) == 2 else buffer_distance
    return cut_line.buffer(buffer_distance)
