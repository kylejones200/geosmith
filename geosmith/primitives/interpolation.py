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
        raise ValueError("Sample and query coordinates must have same dimensionality")

    sample_values = np.asarray(sample_values, dtype=np.float64)
    if len(sample_values) != len(sample_coords):
        raise ValueError("Sample values must have same length as sample coordinates")

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
        query_coords = np.column_stack(
            [query_coords, np.full(len(query_coords), z_mean)]
        )

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


def compute_idw_residuals(
    sample_points: PointSet,
    sample_values: np.ndarray,
    k: int = 16,
    power: float = 2.0,
    max_samples: Optional[int] = 1000,
    leave_one_out: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute IDW predictions and residuals using leave-one-out cross-validation.

    This is useful for hybrid IDW+ML models where ML learns the residuals
    between actual values and IDW predictions.

    Args:
        sample_points: PointSet with sample locations (n_samples, n_dims).
        sample_values: Sample values (n_samples,).
        k: Number of nearest neighbors for IDW, default 16.
        power: IDW exponent, default 2.0.
        max_samples: Maximum number of samples to process (for speed).
                    If None, processes all samples, default 1000.
        leave_one_out: If True, use true leave-one-out (slower but more accurate).
                      If False, use full IDW (faster but biased), default True.

    Returns:
        Tuple of (idw_predictions, residuals):
            - idw_predictions: IDW predictions at sample locations (n_samples,).
            - residuals: Actual - Predicted (n_samples,).

    Example:
        >>> from geosmith import PointSet
        >>> import numpy as np
        >>>
        >>> coords = np.random.rand(100, 3) * 100
        >>> values = np.random.randn(100) * 2 + 10
        >>> samples = PointSet(coordinates=coords)
        >>>
        >>> idw_pred, residuals = compute_idw_residuals(samples, values, k=16)
        >>> print(f"Residual mean: {residuals.mean():.4f}, std: {residuals.std():.4f}")

    Raises:
        ImportError: If scikit-learn is not available.
        ValueError: If inputs are invalid.
    """
    import logging

    logger = logging.getLogger(__name__)

    try:
        from sklearn.neighbors import KDTree
    except ImportError:
        raise ImportError(
            "scikit-learn is required for IDW residuals. "
            "Install with: pip install scikit-learn"
        )

    coords = sample_points.coordinates
    values = np.asarray(sample_values, dtype=np.float64)

    n_samples = len(coords)

    if max_samples and n_samples > max_samples:
        logger.warning(
            f"Processing {max_samples} of {n_samples} samples. "
            "Set max_samples=None for full processing."
        )
        n_process = max_samples
    else:
        n_process = n_samples

    idw_predictions = np.zeros(n_samples, dtype=np.float64)

    if leave_one_out:
        # True leave-one-out (more accurate but slower)
        for i in range(n_process):
            # Leave out sample i
            coords_train = np.delete(coords, i, axis=0)
            values_train = np.delete(values, i)

            # Predict at sample i
            query_coords = coords[i : i + 1]
            query_points = PointSet(coordinates=query_coords)
            train_points = PointSet(coordinates=coords_train)

            idw_predictions[i] = idw_interpolate(
                train_points, values_train, query_points, k=k, power=power
            )[0]

        # For remaining samples, use full IDW (small bias but fast)
        if n_samples > n_process:
            query_points_all = PointSet(coordinates=coords[n_process:])
            idw_predictions[n_process:] = idw_interpolate(
                sample_points, values, query_points_all, k=k, power=power
            )
    else:
        # Use full IDW for all samples (faster but biased)
        query_points_all = PointSet(coordinates=coords)
        idw_predictions = idw_interpolate(
            sample_points, values, query_points_all, k=k, power=power
        )

    # Compute residuals
    residuals = values - idw_predictions

    logger.info(
        f"IDW residuals: mean={residuals.mean():.4f}, "
        f"std={residuals.std():.4f}, "
        f"range=[{residuals.min():.4f}, {residuals.max():.4f}]"
    )

    return idw_predictions, residuals
