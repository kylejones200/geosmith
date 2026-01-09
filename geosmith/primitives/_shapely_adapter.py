"""Optional shapely adapter functions.

These functions provide shapely-backed implementations when available,
with clear error messages when shapely is not installed.
"""

from typing import Tuple

import numpy as np

from geosmith.objects.pointset import PointSet
from geosmith.objects.polygonset import PolygonSet


def haversine_distance(
    coords1: np.ndarray, coords2: np.ndarray
) -> np.ndarray:
    """Compute haversine distance using shapely if available."""
    try:
        from shapely.geometry import Point
        from shapely.ops import transform
        import pyproj
    except ImportError:
        raise ImportError(
            "haversine distance requires shapely and pyproj. "
            "Install with: pip install geosmith[vector]"
        )

    # Implementation would go here
    # For now, raise not implemented
    raise NotImplementedError("Haversine distance not yet implemented")


def point_in_polygon_shapely(
    points: PointSet, polygons: PolygonSet
) -> np.ndarray:
    """Check point in polygon using shapely if available."""
    try:
        from shapely.geometry import Point, Polygon
    except ImportError:
        raise ImportError(
            "point_in_polygon with shapely requires shapely. "
            "Install with: pip install geosmith[vector]"
        )

    n_points = len(points.coordinates)
    n_polygons = len(polygons.rings)
    result = np.zeros((n_points, n_polygons), dtype=bool)

    # Convert points to shapely Points
    shapely_points = [Point(coord) for coord in points.coordinates]

    # Convert polygons to shapely Polygons
    shapely_polygons = []
    for polygon_rings in polygons.rings:
        if not polygon_rings:
            shapely_polygons.append(None)
            continue
        exterior = polygon_rings[0]
        holes = polygon_rings[1:] if len(polygon_rings) > 1 else None
        try:
            poly = Polygon(exterior, holes=holes)
            shapely_polygons.append(poly)
        except Exception:
            shapely_polygons.append(None)

    # Check each point against each polygon
    for i, point in enumerate(shapely_points):
        for j, poly in enumerate(shapely_polygons):
            if poly is not None:
                result[i, j] = poly.contains(point)

    return result

