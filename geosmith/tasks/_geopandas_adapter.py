"""Optional geopandas adapter functions."""

import numpy as np

from geosmith.objects.pointset import PointSet
from geosmith.objects.polygonset import PolygonSet


def buffer_with_geopandas(points: PointSet, distance: float) -> PolygonSet:
    """Buffer points using geopandas if available."""
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError:
        raise ImportError(
            "Buffer with geopandas requires geopandas and shapely. "
            "Install with: pip install geosmith[vector]"
        )

    # Convert to geopandas
    gdf = gpd.GeoDataFrame(
        geometry=[Point(coord) for coord in points.coordinates],
        crs=points.index.crs if points.index else None,
    )

    # Buffer
    buffered = gdf.buffer(distance)

    # Convert back to PolygonSet
    rings = []
    for geom in buffered.geometry:
        if geom.geom_type == "Polygon":
            exterior = np.array(geom.exterior.coords)
            holes = [np.array(ring.coords) for ring in geom.interiors]
            rings.append([exterior] + holes)
        else:
            # MultiPolygon or other - simplify to first polygon
            if hasattr(geom, "geoms"):
                first = geom.geoms[0]
                exterior = np.array(first.exterior.coords)
                rings.append([exterior])

    return PolygonSet(rings=rings, index=points.index)


def buffer_polygons_with_geopandas(polygons: PolygonSet, distance: float) -> PolygonSet:
    """Buffer polygons using geopandas if available."""
    try:
        import geopandas as gpd
        from shapely.geometry import Polygon
    except ImportError:
        raise ImportError(
            "Buffer with geopandas requires geopandas and shapely. "
            "Install with: pip install geosmith[vector]"
        )

    # Convert to geopandas
    shapely_polys = []
    for polygon_rings in polygons.rings:
        if not polygon_rings:
            continue
        exterior = polygon_rings[0]
        holes = polygon_rings[1:] if len(polygon_rings) > 1 else None
        shapely_polys.append(Polygon(exterior, holes=holes))

    gdf = gpd.GeoDataFrame(
        geometry=shapely_polys,
        crs=polygons.index.crs if polygons.index else None,
    )

    # Buffer
    buffered = gdf.buffer(distance)

    # Convert back to PolygonSet
    rings = []
    for geom in buffered.geometry:
        if geom.geom_type == "Polygon":
            exterior = np.array(geom.exterior.coords)
            holes = [np.array(ring.coords) for ring in geom.interiors]
            rings.append([exterior] + holes)

    return PolygonSet(rings=rings, index=polygons.index)

