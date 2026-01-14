"""Data quality tools for geospatial data validation and cleaning.

Provides tools for:
- Outlier detection (spatial outliers)
- Data validation (duplicate samples, coordinate errors)
- Missing data handling (spatial imputation)
- Quality flags (propagate through workflows)
"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from geosmith.objects.pointset import PointSet
from geosmith.utils.errors import DataValidationError
from geosmith.utils.optional_imports import optional_import_single

# Optional dependencies
SCIPY_AVAILABLE, cdist_module = optional_import_single("scipy.spatial.distance", "cdist")
if SCIPY_AVAILABLE:
    from scipy.spatial.distance import cdist  # type: ignore
else:
    cdist = None  # type: ignore

SKLEARN_AVAILABLE, isolation_forest = optional_import_single(
    "sklearn.ensemble", "IsolationForest"
)
if SKLEARN_AVAILABLE:
    from sklearn.ensemble import IsolationForest  # type: ignore
else:
    IsolationForest = None  # type: ignore


@dataclass
class QualityFlags:
    """Container for data quality flags.

    Attributes:
        is_outlier: Boolean array indicating outliers (n_samples,).
        is_duplicate: Boolean array indicating duplicate samples (n_samples,).
        has_invalid_coords: Boolean array indicating invalid coordinates (n_samples,).
        has_missing_value: Boolean array indicating missing values (n_samples,).
        quality_score: Float array with quality scores [0, 1] (n_samples,).
    """

    is_outlier: np.ndarray
    is_duplicate: np.ndarray
    has_invalid_coords: np.ndarray
    has_missing_value: np.ndarray
    quality_score: np.ndarray

    def __post_init__(self) -> None:
        """Validate quality flags."""
        n = len(self.is_outlier)
        if (
            len(self.is_duplicate) != n
            or len(self.has_invalid_coords) != n
            or len(self.has_missing_value) != n
            or len(self.quality_score) != n
        ):
            raise ValueError("All quality flag arrays must have same length")

    def get_good_samples(self) -> np.ndarray:
        """Get indices of samples passing all quality checks.

        Returns:
            Boolean array indicating good samples.
        """
        return ~(
            self.is_outlier
            | self.is_duplicate
            | self.has_invalid_coords
            | self.has_missing_value
        )

    def get_bad_samples(self) -> np.ndarray:
        """Get indices of samples failing any quality check.

        Returns:
            Boolean array indicating bad samples.
        """
        return ~self.get_good_samples()


def detect_spatial_outliers(
    points: PointSet,
    values: np.ndarray,
    method: Literal["isolation_forest", "z_score", "iqr"] = "isolation_forest",
    contamination: float = 0.1,
    z_threshold: float = 3.0,
    iqr_factor: float = 1.5,
) -> np.ndarray:
    """Detect spatial outliers in point data.

    Outliers are points with values that are unusual given their spatial location.

    Args:
        points: PointSet with sample locations.
        values: Sample values (n_samples,).
        method: Outlier detection method:
            - 'isolation_forest': Uses Isolation Forest (requires scikit-learn)
            - 'z_score': Uses Z-score based on spatial neighbors
            - 'iqr': Uses Interquartile Range based on spatial neighbors
        contamination: Expected fraction of outliers (for isolation_forest).
        z_threshold: Z-score threshold (for z_score method).
        iqr_factor: IQR multiplier (for iqr method).

    Returns:
        Boolean array indicating outliers (n_samples,).

    Raises:
        ValueError: If inputs are invalid.
        ImportError: If required dependencies are missing.
    """
    coordinates = points.coordinates

    if len(coordinates) != len(values):
        raise ValueError(
            f"Coordinates ({len(coordinates)}) and values ({len(values)}) "
            f"must have same length"
        )

    if len(values) < 3:
        raise ValueError(f"Need at least 3 samples, got {len(values)}")

    if method == "isolation_forest":
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for isolation_forest method. "
                "Install with: pip install geosmith[ml] or pip install scikit-learn"
            )

        # Combine spatial coordinates and values as features
        features = np.column_stack([coordinates, values])

        # Fit Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination, random_state=42
        )
        outlier_labels = iso_forest.fit_predict(features)

        # Convert to boolean (outliers are -1)
        return outlier_labels == -1

    elif method == "z_score":
        if not SCIPY_AVAILABLE:
            raise ImportError(
                "scipy is required for z_score method. "
                "Install with: pip install geosmith[primitives] or pip install scipy"
            )

        # Compute mean and std of values at nearby locations
        distances = cdist(coordinates, coordinates)
        n_neighbors = min(10, len(values) - 1)

        outliers = np.zeros(len(values), dtype=bool)

        for i in range(len(values)):
            # Find nearest neighbors (excluding self)
            nearest_indices = np.argsort(distances[i])[1 : n_neighbors + 1]
            neighbor_values = values[nearest_indices]

            if len(neighbor_values) > 0:
                mean_val = np.mean(neighbor_values)
                std_val = np.std(neighbor_values)

                if std_val > 1e-9:
                    z_score = abs((values[i] - mean_val) / std_val)
                    outliers[i] = z_score > z_threshold

        return outliers

    elif method == "iqr":
        if not SCIPY_AVAILABLE:
            raise ImportError(
                "scipy is required for iqr method. "
                "Install with: pip install geosmith[primitives] or pip install scipy"
            )

        # Compute IQR of values at nearby locations
        distances = cdist(coordinates, coordinates)
        n_neighbors = min(10, len(values) - 1)

        outliers = np.zeros(len(values), dtype=bool)

        for i in range(len(values)):
            # Find nearest neighbors (excluding self)
            nearest_indices = np.argsort(distances[i])[1 : n_neighbors + 1]
            neighbor_values = values[nearest_indices]

            if len(neighbor_values) > 2:
                q1 = np.percentile(neighbor_values, 25)
                q3 = np.percentile(neighbor_values, 75)
                iqr = q3 - q1

                if iqr > 1e-9:
                    lower_bound = q1 - iqr_factor * iqr
                    upper_bound = q3 + iqr_factor * iqr
                    outliers[i] = (values[i] < lower_bound) | (
                        values[i] > upper_bound
                    )

        return outliers

    else:
        raise ValueError(
            f"Unknown method: {method}. Must be one of: "
            f"['isolation_forest', 'z_score', 'iqr']"
        )


def detect_duplicate_samples(
    points: PointSet,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Detect duplicate samples based on coordinates.

    Args:
        points: PointSet with sample locations.
        tolerance: Distance tolerance for considering samples duplicates (meters).

    Returns:
        Boolean array indicating duplicate samples (n_samples,).
        True means this sample is a duplicate of a previous one.
    """
    coordinates = points.coordinates

    if not SCIPY_AVAILABLE:
        raise ImportError(
            "scipy is required for duplicate detection. "
            "Install with: pip install geosmith[primitives] or pip install scipy"
        )

    n = len(coordinates)
    is_duplicate = np.zeros(n, dtype=bool)

    # For each point, check if it's close to any previous point
    for i in range(1, n):
        distances = cdist(
            coordinates[i : i + 1], coordinates[:i]
        ).ravel()
        if np.any(distances < tolerance):
            is_duplicate[i] = True

    return is_duplicate


def validate_coordinates(
    points: PointSet,
    check_finite: bool = True,
    check_bounds: Optional[dict[str, tuple[float, float]]] = None,
) -> np.ndarray:
    """Validate coordinate data.

    Args:
        points: PointSet with sample locations.
        check_finite: Whether to check for finite (non-NaN, non-inf) values.
        check_bounds: Optional dictionary with bounds to check:
            {'x': (min, max), 'y': (min, max), 'z': (min, max)}

    Returns:
        Boolean array indicating invalid coordinates (n_samples,).
    """
    coordinates = points.coordinates
    n_samples, n_dims = coordinates.shape

    has_invalid = np.zeros(n_samples, dtype=bool)

    # Check for finite values
    if check_finite:
        has_invalid |= ~np.isfinite(coordinates).any(axis=1)

    # Check bounds
    if check_bounds is not None:
        for dim_idx, dim_name in enumerate(["x", "y", "z"][:n_dims]):
            if dim_name in check_bounds:
                min_val, max_val = check_bounds[dim_name]
                has_invalid |= (coordinates[:, dim_idx] < min_val) | (
                    coordinates[:, dim_idx] > max_val
                )

    return has_invalid


def detect_missing_values(values: np.ndarray) -> np.ndarray:
    """Detect missing values in data array.

    Args:
        values: Sample values (n_samples,).

    Returns:
        Boolean array indicating missing values (n_samples,).
    """
    return ~np.isfinite(values)


def compute_quality_flags(
    points: PointSet,
    values: np.ndarray,
    outlier_method: Literal["isolation_forest", "z_score", "iqr"] = "isolation_forest",
    duplicate_tolerance: float = 1e-6,
    check_coords: bool = True,
    check_bounds: Optional[dict[str, tuple[float, float]]] = None,
) -> QualityFlags:
    """Compute comprehensive quality flags for point data.

    Args:
        points: PointSet with sample locations.
        values: Sample values (n_samples,).
        outlier_method: Method for outlier detection.
        duplicate_tolerance: Distance tolerance for duplicate detection.
        check_coords: Whether to validate coordinates.
        check_bounds: Optional bounds for coordinate validation.

    Returns:
        QualityFlags object with all quality checks.
    """
    # Detect outliers
    is_outlier = detect_spatial_outliers(points, values, method=outlier_method)

    # Detect duplicates
    is_duplicate = detect_duplicate_samples(points, tolerance=duplicate_tolerance)

    # Validate coordinates
    has_invalid_coords = (
        validate_coordinates(points, check_bounds=check_bounds)
        if check_coords
        else np.zeros(len(values), dtype=bool)
    )

    # Detect missing values
    has_missing_value = detect_missing_values(values)

    # Compute quality score (0 = bad, 1 = good)
    quality_score = np.ones(len(values))
    quality_score[is_outlier] *= 0.5
    quality_score[is_duplicate] *= 0.3
    quality_score[has_invalid_coords] *= 0.1
    quality_score[has_missing_value] *= 0.0

    return QualityFlags(
        is_outlier=is_outlier,
        is_duplicate=is_duplicate,
        has_invalid_coords=has_invalid_coords,
        has_missing_value=has_missing_value,
        quality_score=quality_score,
    )


def filter_by_quality(
    points: PointSet,
    values: np.ndarray,
    quality_flags: QualityFlags,
    min_quality_score: float = 0.5,
) -> tuple[PointSet, np.ndarray]:
    """Filter points and values based on quality flags.

    Args:
        points: PointSet with sample locations.
        values: Sample values (n_samples,).
        quality_flags: QualityFlags object.
        min_quality_score: Minimum quality score to keep (0-1).

    Returns:
        Tuple of (filtered_points, filtered_values).
    """
    good_mask = quality_flags.quality_score >= min_quality_score

    filtered_coords = points.coordinates[good_mask]
    filtered_values = values[good_mask]

    filtered_points = PointSet(coordinates=filtered_coords)

    return filtered_points, filtered_values


def impute_missing_spatial(
    points: PointSet,
    values: np.ndarray,
    method: Literal["idw", "nearest", "mean"] = "idw",
    k: int = 8,
) -> np.ndarray:
    """Impute missing values using spatial interpolation.

    Args:
        points: PointSet with sample locations.
        values: Sample values (n_samples,) with NaN/inf for missing.
        method: Imputation method:
            - 'idw': Inverse Distance Weighting
            - 'nearest': Nearest neighbor value
            - 'mean': Mean of k nearest neighbors
        k: Number of neighbors to use (for idw and mean methods).

    Returns:
        Imputed values array (n_samples,).
    """
    coordinates = points.coordinates
    imputed_values = values.copy()

    # Find missing values
    missing_mask = ~np.isfinite(values)
    n_missing = np.sum(missing_mask)

    if n_missing == 0:
        return imputed_values

    if not SCIPY_AVAILABLE:
        raise ImportError(
            "scipy is required for spatial imputation. "
            "Install with: pip install geosmith[primitives] or pip install scipy"
        )

    # Get valid samples
    valid_mask = ~missing_mask
    valid_coords = coordinates[valid_mask]
    valid_values = values[valid_mask]

    if len(valid_values) == 0:
        raise DataValidationError(
            "No valid samples available for imputation",
            suggestion="Check data for valid values",
        )

    # Impute each missing value
    missing_coords = coordinates[missing_mask]

    for i, missing_coord in enumerate(missing_coords):
        # Find distances to valid samples
        distances = cdist(missing_coord.reshape(1, -1), valid_coords).ravel()

        # Get k nearest neighbors
        k_actual = min(k, len(valid_values))
        nearest_indices = np.argsort(distances)[:k_actual]
        nearest_distances = distances[nearest_indices]
        nearest_values = valid_values[nearest_indices]

        if method == "idw":
            # Inverse Distance Weighting
            # Avoid division by zero
            nearest_distances = np.maximum(nearest_distances, 1e-9)
            weights = 1.0 / (nearest_distances ** 2)
            weights /= weights.sum()
            imputed_values[np.where(missing_mask)[0][i]] = np.dot(
                weights, nearest_values
            )

        elif method == "nearest":
            # Nearest neighbor
            imputed_values[np.where(missing_mask)[0][i]] = nearest_values[0]

        elif method == "mean":
            # Mean of k nearest neighbors
            imputed_values[np.where(missing_mask)[0][i]] = np.mean(nearest_values)

        else:
            raise ValueError(
                f"Unknown method: {method}. Must be one of: ['idw', 'nearest', 'mean']"
            )

    return imputed_values

