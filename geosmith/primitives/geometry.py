"""Pure geometry operations."""

from typing import Optional, Tuple

import numpy as np

from geosmith.objects.lineset import LineSet
from geosmith.objects.pointset import PointSet
from geosmith.objects.polygonset import PolygonSet


def distance_metric(
    coords1: np.ndarray, coords2: np.ndarray, metric: str = "euclidean"
) -> np.ndarray:
    """Compute distance between two coordinate arrays.

    Args:
        coords1: Array of shape (n1, n_dims).
        coords2: Array of shape (n2, n_dims).
        metric: Distance metric, 'euclidean' or 'haversine'.

    Returns:
        Array of shape (n1, n2) with pairwise distances.
    """
    if metric == "euclidean":
        diff = coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :]
        return np.sqrt(np.sum(diff**2, axis=2))
    elif metric == "haversine":
        # Haversine distance for lat/lon coordinates
        try:
            from geosmith.primitives._shapely_adapter import haversine_distance
        except ImportError:
            raise ImportError(
                "haversine distance requires shapely. "
                "Install with: pip install geosmith[vector]"
            )
        return haversine_distance(coords1, coords2)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def nearest_neighbor_search(
    query_points: PointSet, target_points: PointSet, k: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Find k nearest neighbors.

    Args:
        query_points: PointSet to query.
        target_points: PointSet to search in.
        k: Number of neighbors to find.

    Returns:
        Tuple of (indices, distances) arrays, each of shape (n_query, k).
    """
    distances = distance_metric(
        query_points.coordinates, target_points.coordinates, metric="euclidean"
    )

    # Get k nearest
    k = min(k, len(target_points.coordinates))
    indices = np.argsort(distances, axis=1)[:, :k]
    dists = np.take_along_axis(distances, indices, axis=1)

    return indices, dists


def point_in_polygon(points: PointSet, polygons: PolygonSet) -> np.ndarray:
    """Check which points are inside which polygons.

    Args:
        points: PointSet to check.
        polygons: PolygonSet to check against.

    Returns:
        Boolean array of shape (n_points, n_polygons).
    """
    try:
        from geosmith.primitives._shapely_adapter import point_in_polygon_shapely
    except ImportError:
        # Fallback to simple bounding box check
        return point_in_polygon_bbox(points, polygons)

    return point_in_polygon_shapely(points, polygons)


def point_in_polygon_bbox(points: PointSet, polygons: PolygonSet) -> np.ndarray:
    """Simple bounding box check (fallback when shapely not available).

    Args:
        points: PointSet to check.
        polygons: PolygonSet to check against.

    Returns:
        Boolean array of shape (n_points, n_polygons).
    """
    n_points = len(points.coordinates)
    n_polygons = len(polygons.rings)
    result = np.zeros((n_points, n_polygons), dtype=bool)

    for poly_idx, polygon_rings in enumerate(polygons.rings):
        if not polygon_rings:
            continue
        exterior = polygon_rings[0]
        minx, miny = exterior.min(axis=0)[:2]
        maxx, maxy = exterior.max(axis=0)[:2]

        point_coords = points.coordinates[:, :2]
        in_bbox = (
            (point_coords[:, 0] >= minx)
            & (point_coords[:, 0] <= maxx)
            & (point_coords[:, 1] >= miny)
            & (point_coords[:, 1] <= maxy)
        )
        result[:, poly_idx] = in_bbox

    return result


def line_length(line: LineSet, line_idx: int = 0) -> float:
    """Compute total length of a line.

    Args:
        line: LineSet containing the line.
        line_idx: Index of the line to measure.

    Returns:
        Total length of the line.
    """
    vertices = line.vertices[line_idx]
    if len(vertices) < 2:
        return 0.0

    diffs = np.diff(vertices, axis=0)
    lengths = np.sqrt(np.sum(diffs**2, axis=1))
    return float(np.sum(lengths))


def polygon_area(polygon: PolygonSet, poly_idx: int = 0) -> float:
    """Compute area of a polygon using shoelace formula.

    Args:
        polygon: PolygonSet containing the polygon.
        poly_idx: Index of the polygon to measure.

    Returns:
        Area of the polygon (exterior ring only, holes not subtracted).
    """
    rings = polygon.rings[poly_idx]
    if not rings:
        return 0.0

    exterior = rings[0]
    if len(exterior) < 3:
        return 0.0

    # Shoelace formula for 2D
    x = exterior[:, 0]
    y = exterior[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return float(area)


def bounding_box(
    points: Optional[PointSet] = None,
    lines: Optional[LineSet] = None,
    polygons: Optional[PolygonSet] = None,
) -> Tuple[float, float, float, float]:
    """Compute bounding box of geometry objects.

    Args:
        points: Optional PointSet.
        lines: Optional LineSet.
        polygons: Optional PolygonSet.

    Returns:
        Bounding box as (minx, miny, maxx, maxy).
    """
    all_coords = []

    if points is not None:
        all_coords.append(points.coordinates[:, :2])

    if lines is not None:
        for verts in lines.vertices:
            all_coords.append(verts[:, :2])

    if polygons is not None:
        for polygon_rings in polygons.rings:
            for ring in polygon_rings:
                all_coords.append(ring[:, :2])

    if not all_coords:
        raise ValueError("At least one geometry object must be provided")

    combined = np.vstack(all_coords)
    minx, miny = combined.min(axis=0)
    maxx, maxy = combined.max(axis=0)

    return (float(minx), float(miny), float(maxx), float(maxy))
