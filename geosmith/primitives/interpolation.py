"""Spatial interpolation primitives.

Pure interpolation operations that work with Layer 1 objects.
"""

from typing import Optional

import numpy as np

from geosmith.objects.pointset import PointSet
from geosmith.objects.rastergrid import RasterGrid


def idw_interpolate(
    sample_points: PointSet,
    sample_values: np.ndarray,
    query_points: PointSet,
    k: int = 16,
    power: float = 2.0,
    eps: float = 1e-9,
) -> np.ndarray:
    """Inverse Distance Weighted (IDW) interpolation.

    Estimates values at query locations using weighted average of k nearest
    neighbors, with weights inversely proportional to distance raised to power.

    Args:
        sample_points: PointSet with sample locations (n_samples, n_dims).
        sample_values: Sample values (n_samples,) - e.g., ore grades, measurements.
        query_points: PointSet with query locations (n_queries, n_dims).
        k: Number of nearest neighbors to use.
        power: IDW exponent (typically 2.0). Higher values give more weight
               to closer points.
        eps: Minimum distance to avoid division by zero.

    Returns:
        Estimated values at query points (n_queries,).

    Raises:
        ImportError: If scikit-learn is not available.
        ValueError: If inputs are invalid.

    Example:
        >>> from geosmith import PointSet, GeoIndex
        >>> import numpy as np
        >>> 
        >>> # Sample locations and grades
        >>> sample_coords = np.array([[100, 200, 50], [150, 250, 60], [120, 230, 55]])
        >>> sample_values = np.array([2.5, 1.8, 2.1])
        >>> samples = PointSet(coordinates=sample_coords)
        >>> 
        >>> # Query point
        >>> query_coords = np.array([[130, 220, 57]])
        >>> queries = PointSet(coordinates=query_coords)
        >>> 
        >>> # Interpolate
        >>> grade = idw_interpolate(samples, sample_values, queries, k=3, power=2.0)
        >>> print(f"Estimated grade: {grade[0]:.2f}")
    """
    try:
        from sklearn.neighbors import KDTree
    except ImportError:
        raise ImportError(
            "IDW interpolation requires scikit-learn. "
            "Install with: pip install scikit-learn"
        )

    sample_coords = sample_points.coordinates
    query_coords = query_points.coordinates

    if sample_coords.ndim != 2 or query_coords.ndim != 2:
        raise ValueError("Coordinates must be 2D arrays (n_points, n_dim)")

    if sample_coords.shape[1] != query_coords.shape[1]:
        raise ValueError(
            "Sample and query coordinates must have same dimensionality"
        )

    sample_values = np.asarray(sample_values, dtype=np.float64)
    if len(sample_values) != len(sample_coords):
        raise ValueError(
            "Sample values must have same length as sample coordinates"
        )

    if len(sample_coords) == 0:
        raise ValueError("Sample coordinates cannot be empty")

    # Build KDTree for efficient nearest neighbor search
    tree = KDTree(sample_coords)

    # Find k nearest neighbors (or all neighbors if k > n_samples)
    k_actual = min(k, len(sample_coords))
    distances, indices = tree.query(query_coords, k=k_actual)

    # Handle case where query returns 1D arrays (single query point, k=1)
    if distances.ndim == 1:
        distances = distances.reshape(-1, 1)
        indices = indices.reshape(-1, 1)

    # Compute IDW weights: w = 1 / distance^power
    weights = 1.0 / np.maximum(distances, eps) ** power

    # Normalize weights to sum to 1 for each query point
    weights /= weights.sum(axis=1, keepdims=True)

    # Weighted average: sum(weight * value) for each query point
    estimates = (sample_values[indices] * weights).sum(axis=1)

    return estimates


def idw_to_raster(
    sample_points: PointSet,
    sample_values: np.ndarray,
    target_raster: RasterGrid,
    k: int = 16,
    power: float = 2.0,
    eps: float = 1e-9,
) -> RasterGrid:
    """Interpolate point values to raster grid using IDW.

    Args:
        sample_points: PointSet with sample locations.
        sample_values: Sample values (n_samples,).
        target_raster: Target RasterGrid to interpolate into.
        k: Number of nearest neighbors.
        power: IDW exponent.
        eps: Minimum distance.

    Returns:
        RasterGrid with interpolated values.
    """
    # Extract query coordinates from raster grid
    # This is a simplified version - production would properly handle affine transform
    if target_raster.data.ndim == 2:
        n_rows, n_cols = target_raster.data.shape
        n_bands = 1
    else:
        n_bands, n_rows, n_cols = target_raster.data.shape

    # Generate query coordinates from raster transform
    # For now, create a simple grid (production would use proper transform)
    a, b, c, d, e, f = target_raster.transform
    x_coords = np.arange(n_cols) * abs(a) + c
    y_coords = np.arange(n_rows) * abs(e) + f

    X, Y = np.meshgrid(x_coords, y_coords)
    query_coords = np.column_stack([X.ravel(), Y.ravel()])

    # Add Z coordinate if samples are 3D
    if sample_points.coordinates.shape[1] == 3:
        # Use mean Z for 2D raster
        z_mean = sample_points.coordinates[:, 2].mean()
        query_coords = np.column_stack([query_coords, np.full(len(query_coords), z_mean)])

    query_points = PointSet(coordinates=query_coords)

    # Interpolate
    interpolated = idw_interpolate(
        sample_points, sample_values, query_points, k=k, power=power, eps=eps
    )

    # Reshape to raster shape
    if n_bands == 1:
        raster_data = interpolated.reshape(n_rows, n_cols)
    else:
        raster_data = interpolated.reshape(n_bands, n_rows, n_cols)

    return RasterGrid(
        data=raster_data,
        transform=target_raster.transform,
        nodata=target_raster.nodata,
        band_names=target_raster.band_names,
        index=target_raster.index,
    )

