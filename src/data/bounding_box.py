from shapely.geometry import mapping, shape, Polygon, box
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


def cut_linestrings_at_bounds(bounds, linestring):
    lineobj = wkt.loads(str(shape(linestring)))
    conus_transformed_poly = wkt.loads(str(bounds))
    conus_intersection = conus_transformed_poly.intersection(lineobj)
    print("mapping(conus_intersection)")
    print(mapping(conus_intersection))
    s = shape(mapping(conus_intersection))
    print("done mapping(conus_intersection)")
    return s
