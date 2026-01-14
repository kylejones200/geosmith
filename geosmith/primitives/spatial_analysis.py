"""Spatial analysis tools: autocorrelation, hotspot detection, and clustering.

Provides tools for:
- Spatial autocorrelation (Moran's I, Geary's C)
- Hotspot detection (Getis-Ord Gi*)
- Spatial weights generation
"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from geosmith.objects.pointset import PointSet
from geosmith.objects.polygonset import PolygonSet
from geosmith.utils.optional_imports import optional_import_single

SCIPY_AVAILABLE, _ = optional_import_single("scipy.spatial.distance", "cdist")
if SCIPY_AVAILABLE:
    from scipy.spatial.distance import cdist  # type: ignore
else:
    cdist = None  # type: ignore


@dataclass
class SpatialWeights:
    """Spatial weights matrix for spatial autocorrelation analysis.

    Attributes:
        weights: Sparse or dense weights matrix (n x n).
        neighbors: Dictionary mapping index to list of neighbor indices.
        n_observations: Number of observations.
        weights_type: Type of weights ('queen', 'rook', 'knn', 'distance').
    """

    weights: np.ndarray
    neighbors: dict[int, list[int]]
    n_observations: int
    weights_type: str

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SpatialWeights(type={self.weights_type}, "
            f"n={self.n_observations}, "
            f"avg_neighbors={np.mean([len(v) for v in self.neighbors.values()]):.1f})"
        )


@dataclass
class MoranResult:
    """Results from Moran's I spatial autocorrelation test.

    Attributes:
        I: Moran's I statistic (range: -1 to 1).
        z_score: Standardized z-score.
        p_value: P-value for significance test.
        expected_I: Expected value under null hypothesis (typically near 0).
        variance: Variance of I under null hypothesis.
    """

    I: float
    z_score: float
    p_value: float
    expected_I: float
    variance: float

    def __repr__(self) -> str:
        """String representation."""
        significance = "***" if self.p_value < 0.001 else "**" if self.p_value < 0.01 else "*" if self.p_value < 0.05 else ""
        return (
            f"MoranResult(I={self.I:.4f}, z={self.z_score:.2f}, "
            f"p={self.p_value:.4f}{significance})"
        )


@dataclass
class GearyResult:
    """Results from Geary's C spatial autocorrelation test.

    Attributes:
        C: Geary's C statistic (range: 0 to 2, typically 0.5 to 1.5).
        z_score: Standardized z-score.
        p_value: P-value for significance test.
        expected_C: Expected value under null hypothesis (typically 1.0).
        variance: Variance of C under null hypothesis.
    """

    C: float
    z_score: float
    p_value: float
    expected_C: float
    variance: float

    def __repr__(self) -> str:
        """String representation."""
        significance = "***" if self.p_value < 0.001 else "**" if self.p_value < 0.01 else "*" if self.p_value < 0.05 else ""
        return (
            f"GearyResult(C={self.C:.4f}, z={self.z_score:.2f}, "
            f"p={self.p_value:.4f}{significance})"
        )


@dataclass
class HotspotResult:
    """Results from Getis-Ord Gi* hotspot detection.

    Attributes:
        gi_star: Getis-Ord Gi* statistic for each location.
        z_scores: Standardized z-scores.
        p_values: P-values for significance test.
        hotspots: Boolean array indicating significant hotspots (p < 0.05).
        coldspots: Boolean array indicating significant coldspots (p < 0.05).
    """

    gi_star: np.ndarray
    z_scores: np.ndarray
    p_values: np.ndarray
    hotspots: np.ndarray
    coldspots: np.ndarray

    def __repr__(self) -> str:
        """String representation."""
        n_hot = self.hotspots.sum()
        n_cold = self.coldspots.sum()
        return (
            f"HotspotResult(n_hotspots={n_hot}, n_coldspots={n_cold}, "
            f"mean_gi={self.gi_star.mean():.4f})"
        )


def create_queen_weights(
    polygons: PolygonSet,
) -> SpatialWeights:
    """Create Queen contiguity spatial weights (polygons share vertex or edge).

    Args:
        polygons: PolygonSet with polygon geometries.

    Returns:
        SpatialWeights object with Queen contiguity weights.

    Note:
        This is a simplified implementation. For production use with complex
        geometries, consider using libpysal or geopandas.
    """
    n = len(polygons.rings)
    weights = np.zeros((n, n))
    neighbors = {i: [] for i in range(n)}

    # For each pair of polygons, check if they share a vertex or edge
    # This is simplified - full implementation would use proper geometry intersection
    for i in range(n):
        for j in range(i + 1, n):
            # Check if polygons are neighbors (simplified check)
            # In practice, would use shapely intersection
            if _polygons_adjacent(polygons.rings[i], polygons.rings[j]):
                weights[i, j] = 1.0
                weights[j, i] = 1.0
                neighbors[i].append(j)
                neighbors[j].append(i)

    # Row-standardize
    row_sums = weights.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    weights = weights / row_sums[:, np.newaxis]

    return SpatialWeights(
        weights=weights,
        neighbors=neighbors,
        n_observations=n,
        weights_type="queen",
    )


def create_knn_weights(
    points: PointSet,
    k: int = 8,
) -> SpatialWeights:
    """Create K-nearest neighbors spatial weights.

    Args:
        points: PointSet with point locations.
        k: Number of nearest neighbors (default: 8).

    Returns:
        SpatialWeights object with KNN weights.
    """
    if not SCIPY_AVAILABLE:
        raise ImportError(
            "scipy is required for KNN weights. Install with: pip install scipy"
        )

    coordinates = points.coordinates
    n = len(coordinates)

    if k >= n:
        raise ValueError(f"k ({k}) must be less than number of points ({n})")

    # Compute distance matrix
    distances = cdist(coordinates, coordinates)

    # Set diagonal to infinity to exclude self
    np.fill_diagonal(distances, np.inf)

    # Find k nearest neighbors for each point
    weights = np.zeros((n, n))
    neighbors = {}

    for i in range(n):
        # Get k nearest neighbors
        nearest_indices = np.argsort(distances[i])[:k]
        neighbors[i] = nearest_indices.tolist()

        # Set weights (binary or distance-weighted)
        for j in nearest_indices:
            weights[i, j] = 1.0

    # Row-standardize
    row_sums = weights.sum(axis=1)
    weights = weights / row_sums[:, np.newaxis]

    return SpatialWeights(
        weights=weights,
        neighbors=neighbors,
        n_observations=n,
        weights_type=f"knn_{k}",
    )


def create_distance_weights(
    points: PointSet,
    threshold: float,
    power: float = 1.0,
) -> SpatialWeights:
    """Create distance-based spatial weights.

    Args:
        points: PointSet with point locations.
        threshold: Maximum distance for neighbors.
        power: Power for distance weighting (1 = inverse distance, 2 = inverse squared).

    Returns:
        SpatialWeights object with distance-based weights.
    """
    if not SCIPY_AVAILABLE:
        raise ImportError(
            "scipy is required for distance weights. Install with: pip install scipy"
        )

    coordinates = points.coordinates
    n = len(coordinates)

    # Compute distance matrix
    distances = cdist(coordinates, coordinates)

    # Create weights: inverse distance for neighbors within threshold
    weights = np.zeros((n, n))
    neighbors = {}

    for i in range(n):
        neighbors[i] = []
        for j in range(n):
            if i != j and distances[i, j] <= threshold:
                # Inverse distance weighting
                weight = 1.0 / (distances[i, j] ** power)
                weights[i, j] = weight
                neighbors[i].append(j)

    # Row-standardize
    row_sums = weights.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    weights = weights / row_sums[:, np.newaxis]

    return SpatialWeights(
        weights=weights,
        neighbors=neighbors,
        n_observations=n,
        weights_type=f"distance_{threshold}",
    )


def morans_i(
    values: np.ndarray,
    weights: SpatialWeights,
) -> MoranResult:
    """Compute Moran's I statistic for spatial autocorrelation.

    Moran's I measures spatial autocorrelation:
    - I > 0: Positive autocorrelation (similar values cluster)
    - I < 0: Negative autocorrelation (dissimilar values cluster)
    - I ≈ 0: No spatial autocorrelation

    Args:
        values: Array of values (n_observations,).
        weights: SpatialWeights object.

    Returns:
        MoranResult with statistic, z-score, and p-value.

    Example:
        >>> from geosmith import PointSet
        >>> from geosmith.primitives.spatial_analysis import create_knn_weights, morans_i
        >>>
        >>> points = PointSet(coordinates=coords)
        >>> weights = create_knn_weights(points, k=8)
        >>> result = morans_i(values, weights)
        >>> print(f"Moran's I: {result.I:.4f}, p-value: {result.p_value:.4f}")
    """
    n = len(values)
    if n != weights.n_observations:
        raise ValueError(
            f"values length ({n}) must match weights n_observations ({weights.n_observations})"
        )

    # Center values
    values_centered = values - np.mean(values)
    variance = np.var(values)

    if variance == 0:
        # All values are the same
        return MoranResult(
            I=0.0, z_score=0.0, p_value=1.0, expected_I=-1.0 / (n - 1), variance=0.0
        )

    # Compute Moran's I
    numerator = 0.0
    denominator = 0.0
    w_sum = 0.0

    for i in range(n):
        for j in range(n):
            if i != j:
                w_ij = weights.weights[i, j]
                w_sum += w_ij
                numerator += w_ij * values_centered[i] * values_centered[j]
        denominator += values_centered[i] ** 2

    if w_sum == 0:
        raise ValueError("Sum of weights is zero - no spatial relationships")

    I = (n / w_sum) * (numerator / denominator)

    # Compute expected value and variance under null hypothesis
    expected_I = -1.0 / (n - 1)

    # Variance calculation (simplified - full formula is more complex)
    w_squared_sum = np.sum(weights.weights ** 2)
    w_row_sums = weights.weights.sum(axis=1)
    w_col_sums = weights.weights.sum(axis=0)

    S0 = w_sum
    S1 = 0.5 * np.sum((weights.weights + weights.weights.T) ** 2)
    S2 = np.sum((w_row_sums + w_col_sums) ** 2)

    variance = (
        (n * ((n ** 2 - 3 * n + 3) * S1 - n * S2 + 3 * S0 ** 2))
        - (S1 * (n ** 2 - n) - 2 * n * S2 + 6 * S0 ** 2)
    ) / ((n - 1) * (n - 2) * (n - 3) * S0 ** 2) - expected_I ** 2

    # Z-score and p-value
    if variance <= 0:
        z_score = 0.0
        p_value = 1.0
    else:
        z_score = (I - expected_I) / np.sqrt(variance)
        # Two-tailed test
        try:
            from scipy import stats

            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        except ImportError:
            # Fallback if scipy not available (approximate)
            p_value = 2 * (1 - 0.5 * (1 + np.sign(z_score) * (1 - np.exp(-2 * z_score ** 2 / np.pi)) ** 0.5))

    return MoranResult(
        I=I, z_score=z_score, p_value=p_value, expected_I=expected_I, variance=variance
    )


def gearys_c(
    values: np.ndarray,
    weights: SpatialWeights,
) -> GearyResult:
    """Compute Geary's C statistic for spatial autocorrelation.

    Geary's C is similar to Moran's I but more sensitive to local differences:
    - C < 1: Positive autocorrelation (similar values cluster)
    - C > 1: Negative autocorrelation (dissimilar values cluster)
    - C ≈ 1: No spatial autocorrelation

    Args:
        values: Array of values (n_observations,).
        weights: SpatialWeights object.

    Returns:
        GearyResult with statistic, z-score, and p-value.
    """
    n = len(values)
    if n != weights.n_observations:
        raise ValueError(
            f"values length ({n}) must match weights n_observations ({weights.n_observations})"
        )

    # Compute Geary's C
    numerator = 0.0
    denominator = 0.0
    w_sum = 0.0

    variance = np.var(values)
    if variance == 0:
        return GearyResult(
            C=1.0, z_score=0.0, p_value=1.0, expected_C=1.0, variance=0.0
        )

    for i in range(n):
        for j in range(n):
            if i != j:
                w_ij = weights.weights[i, j]
                w_sum += w_ij
                numerator += w_ij * (values[i] - values[j]) ** 2
        denominator += (values[i] - np.mean(values)) ** 2

    if w_sum == 0:
        raise ValueError("Sum of weights is zero - no spatial relationships")

    C = ((n - 1) / (2 * w_sum)) * (numerator / denominator)

    # Expected value and variance
    expected_C = 1.0

    # Simplified variance calculation
    w_squared_sum = np.sum(weights.weights ** 2)
    w_row_sums = weights.weights.sum(axis=1)

    S0 = w_sum
    S1 = 0.5 * np.sum((weights.weights + weights.weights.T) ** 2)
    S2 = np.sum((w_row_sums + weights.weights.sum(axis=0)) ** 2)

    variance = (
        ((n - 1) * S1 * (n ** 2 - 3 * n + 3 - (n - 1) * S2 / S0))
        - (0.25 * (n - 1) * S2 * (n ** 2 + 3 * n - 6 - (n ** 2 - n + 2) * S2 / S0))
        + (w_squared_sum * (n ** 2 - 3 - (n - 1) ** 2 * S2 / S0))
    ) / (n * (n - 2) * (n - 3) * S0 ** 2)

    # Z-score and p-value
    if variance <= 0:
        z_score = 0.0
        p_value = 1.0
    else:
        z_score = (C - expected_C) / np.sqrt(variance)
        try:
            from scipy import stats

            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        except ImportError:
            # Fallback if scipy not available (approximate)
            p_value = 2 * (1 - 0.5 * (1 + np.sign(z_score) * (1 - np.exp(-2 * z_score ** 2 / np.pi)) ** 0.5))

    return GearyResult(
        C=C, z_score=z_score, p_value=p_value, expected_C=expected_C, variance=variance
    )


def getis_ord_gi_star(
    values: np.ndarray,
    weights: SpatialWeights,
) -> HotspotResult:
    """Compute Getis-Ord Gi* statistic for hotspot detection.

    Identifies statistically significant spatial clusters of high values (hotspots)
    and low values (coldspots).

    Args:
        values: Array of values (n_observations,).
        weights: SpatialWeights object.

    Returns:
        HotspotResult with Gi* statistics, z-scores, p-values, and hotspot/coldspot flags.

    Example:
        >>> from geosmith.primitives.spatial_analysis import getis_ord_gi_star
        >>>
        >>> result = getis_ord_gi_star(values, weights)
        >>> print(f"Hotspots: {result.hotspots.sum()}, Coldspots: {result.coldspots.sum()}")
    """
    n = len(values)
    if n != weights.n_observations:
        raise ValueError(
            f"values length ({n}) must match weights n_observations ({weights.n_observations})"
        )

    mean_val = np.mean(values)
    std_val = np.std(values)

    if std_val == 0:
        # All values are the same
        return HotspotResult(
            gi_star=np.zeros(n),
            z_scores=np.zeros(n),
            p_values=np.ones(n),
            hotspots=np.zeros(n, dtype=bool),
            coldspots=np.zeros(n, dtype=bool),
        )

    # Compute Gi* for each location
    gi_star = np.zeros(n)
    w_sums = np.zeros(n)
    w_squared_sums = np.zeros(n)

    for i in range(n):
        w_sum = 0.0
        w_sq_sum = 0.0
        weighted_sum = 0.0

        for j in range(n):
            w_ij = weights.weights[i, j]
            w_sum += w_ij
            w_sq_sum += w_ij ** 2
            weighted_sum += w_ij * values[j]

        w_sums[i] = w_sum
        w_squared_sums[i] = w_sq_sum
        gi_star[i] = weighted_sum / w_sum

    # Standardize
    S = np.sqrt((n * w_squared_sums - w_sums ** 2) / (n - 1))
    z_scores = (gi_star - mean_val) / (std_val * S / w_sums)

    # P-values (two-tailed)
    try:
        from scipy import stats

        p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
    except ImportError:
        # Fallback if scipy not available (approximate)
        p_values = 2 * (1 - 0.5 * (1 + np.sign(z_scores) * (1 - np.exp(-2 * z_scores ** 2 / np.pi)) ** 0.5))

    # Identify hotspots (high values) and coldspots (low values)
    alpha = 0.05
    hotspots = (z_scores > 0) & (p_values < alpha)
    coldspots = (z_scores < 0) & (p_values < alpha)

    return HotspotResult(
        gi_star=gi_star,
        z_scores=z_scores,
        p_values=p_values,
        hotspots=hotspots,
        coldspots=coldspots,
    )


def _polygons_adjacent(
    rings1: list[np.ndarray], rings2: list[np.ndarray]
) -> bool:
    """Check if two polygons are adjacent (simplified check).

    This is a placeholder - full implementation would use proper geometry
    intersection with shapely.

    Args:
        rings1: Rings for first polygon.
        rings2: Rings for second polygon.

    Returns:
        True if polygons are adjacent, False otherwise.
    """
    # Simplified: check if bounding boxes overlap significantly
    # Full implementation would check actual geometry intersection
    if not rings1 or not rings2:
        return False

    # Get bounding boxes
    coords1 = np.vstack(rings1[0]) if rings1[0] is not None else None
    coords2 = np.vstack(rings2[0]) if rings2[0] is not None else None

    if coords1 is None or coords2 is None:
        return False

    bbox1 = [coords1[:, 0].min(), coords1[:, 1].min(), coords1[:, 0].max(), coords1[:, 1].max()]
    bbox2 = [coords2[:, 0].min(), coords2[:, 1].min(), coords2[:, 0].max(), coords2[:, 1].max()]

    # Check if bounding boxes overlap
    return not (
        bbox1[2] < bbox2[0] or bbox2[2] < bbox1[0] or bbox1[3] < bbox2[1] or bbox2[3] < bbox1[1]
    )

