from shapely.geometry import mapping, shape, Polygon, box, GeometryCollection
from shapely import wkt

import geopandas as gp
import fiona


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
    bound.crs = ({'init': 'epsg:4326'})
    return bound


def bounds2polygon(metafile_data):
    bounds = gp.GeoDataFrame(geometry=[Polygon(metafile_data)])
    bounds.crs = {'init': 'epsg:4326'}
    return bounds


def window_trueBoundingBox(windowBox, imageBox):
    gdf_WindowBounds = bounds2box(windowBox)
    gdf_ImageBounds = bounds2polygon(imageBox)

    gdf_TrueBounds = gp.GeoDataFrame(gp.overlay(gdf_WindowBounds, gdf_ImageBounds, how='intersection').geometry)
    gdf_TrueBounds.crs = ({'init': 'epsg:4326'})
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
            if not shape(mapping(conus_intersection)).is_empty and shape(mapping(conus_intersection)).is_valid:
                geom.append(shape(mapping(conus_intersection)))
                iden.append(label)

    gpd = gp.GeoDataFrame({'label': iden}, geometry=geom, crs={'init': 'epsg:4326'})
    # For testing if different colors appear in the rasterized labels
    # if len(gpd) > 3:
    #     gpd.loc['label'] = 1
    return gpd
