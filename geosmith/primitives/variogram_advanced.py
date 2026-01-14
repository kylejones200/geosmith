"""Advanced variogram models: nested structures and anisotropy.

Provides support for:
- Nested variogram structures (multiple ranges)
- Anisotropic models (directional variograms)
- Zonal anisotropy (different ranges in different directions)
"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from geosmith.objects.pointset import PointSet
from geosmith.primitives.variogram import (
    VariogramModel,
    compute_experimental_variogram,
    fit_variogram_model,
    predict_variogram,
)


@dataclass(frozen=True)
class NestedVariogramModel:
    """Nested variogram structure with multiple components.

    Common in mining applications where multiple scales of variability exist:
    - Nugget (measurement error, micro-scale)
    - Short-range structure (local variability)
    - Long-range structure (regional trends)

    Attributes:
        components: List of VariogramModel components (ordered from short to long range).
    """

    components: tuple[VariogramModel, ...]

    def __post_init__(self) -> None:
        """Validate nested variogram."""
        if len(self.components) == 0:
            raise ValueError("Must have at least one component")
        if len(self.components) > 5:
            raise ValueError("Too many components (max 5)")

    def predict(self, distances: np.ndarray) -> np.ndarray:
        """Predict nested variogram at given distances.

        Sums contributions from all components.

        Args:
            distances: Array of distances.

        Returns:
            Semi-variance values.
        """
        gamma = np.zeros_like(distances)
        for component in self.components:
            gamma += predict_variogram(component, distances)
        return gamma

    @property
    def nugget(self) -> float:
        """Total nugget effect (sum of all component nuggets)."""
        return sum(comp.nugget for comp in self.components)

    @property
    def sill(self) -> float:
        """Total sill (sum of all component sills)."""
        return sum(comp.sill for comp in self.components)

    @property
    def range_param(self) -> float:
        """Maximum range (largest range among components)."""
        return max(comp.range_param for comp in self.components)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"NestedVariogramModel(n_components={len(self.components)}, "
            f"nugget={self.nugget:.4f}, sill={self.sill:.4f}, "
            f"max_range={self.range_param:.4f})"
        )


@dataclass(frozen=True)
class AnisotropicVariogramModel:
    """Anisotropic variogram with directional ranges.

    Accounts for different correlation ranges in different directions.
    Common when geological structures have preferred orientations.

    Attributes:
        base_model: Base variogram model (isotropic).
        anisotropy_ratio: Ratio of major/minor axis ranges.
        anisotropy_angle: Rotation angle in degrees (0 = east, counterclockwise).
        major_range: Range in major direction.
        minor_range: Range in minor direction (major_range / anisotropy_ratio).
    """

    base_model: VariogramModel
    anisotropy_ratio: float
    anisotropy_angle: float
    major_range: float

    def __post_init__(self) -> None:
        """Validate anisotropic variogram."""
        if self.anisotropy_ratio < 1.0:
            raise ValueError(f"anisotropy_ratio must be >= 1.0, got {self.anisotropy_ratio}")
        if self.major_range <= 0:
            raise ValueError(f"major_range must be positive, got {self.major_range}")

    @property
    def minor_range(self) -> float:
        """Range in minor direction."""
        return self.major_range / self.anisotropy_ratio

    def _transform_coordinates(
        self, coordinates: np.ndarray, center: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Transform coordinates to account for anisotropy.

        Rotates and scales coordinates to make variogram isotropic.

        Args:
            coordinates: Array of shape (n_points, n_dims).
            center: Optional center point for rotation (default: origin).

        Returns:
            Transformed coordinates.
        """
        if coordinates.shape[1] != 2:
            raise ValueError("Anisotropy currently only supported for 2D")

        if center is None:
            center = np.zeros(2)

        # Translate to center
        coords_centered = coordinates - center

        # Rotation matrix (negative angle to align major axis with x)
        angle_rad = np.deg2rad(-self.anisotropy_angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        # Rotate
        coords_rotated = coords_centered @ rotation.T

        # Scale by anisotropy ratio (compress minor axis)
        coords_scaled = coords_rotated.copy()
        coords_scaled[:, 1] /= self.anisotropy_ratio

        return coords_scaled

    def predict(
        self,
        distances: np.ndarray,
        direction: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict anisotropic variogram.

        Args:
            distances: Array of distances (isotropic case).
            direction: Optional array of direction vectors (n_points, 2) for directional prediction.

        Returns:
            Semi-variance values.
        """
        # For isotropic distances, use base model with adjusted range
        # The effective range depends on direction
        if direction is None:
            # Use average range (geometric mean)
            effective_range = np.sqrt(self.major_range * self.minor_range)
            # Create temporary model with effective range
            temp_model = VariogramModel(
                model_type=self.base_model.model_type,
                nugget=self.base_model.nugget,
                sill=self.base_model.sill,
                range_param=effective_range,
                partial_sill=self.base_model.partial_sill,
                r_squared=self.base_model.r_squared,
            )
            return predict_variogram(temp_model, distances)
        else:
            # Directional prediction
            # Project direction onto major/minor axes
            angle_rad = np.deg2rad(self.anisotropy_angle)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)

            # Direction vectors
            dir_norm = direction / (np.linalg.norm(direction, axis=1, keepdims=True) + 1e-10)
            major_dir = np.array([cos_a, sin_a])
            minor_dir = np.array([-sin_a, cos_a])

            # Projections
            proj_major = np.abs(dir_norm @ major_dir)
            proj_minor = np.abs(dir_norm @ minor_dir)

            # Effective range in this direction
            effective_range = (
                self.major_range * proj_major + self.minor_range * proj_minor
            )

            # Predict using effective range
            gamma = np.zeros(len(distances))
            for i, (dist, eff_range) in enumerate(zip(distances, effective_range)):
                temp_model = VariogramModel(
                    model_type=self.base_model.model_type,
                    nugget=self.base_model.nugget,
                    sill=self.base_model.sill,
                    range_param=eff_range,
                    partial_sill=self.base_model.partial_sill,
                    r_squared=self.base_model.r_squared,
                )
                gamma[i] = predict_variogram(temp_model, np.array([dist]))[0]

            return gamma

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"AnisotropicVariogramModel(type={self.base_model.model_type}, "
            f"ratio={self.anisotropy_ratio:.2f}, angle={self.anisotropy_angle:.1f}°, "
            f"major_range={self.major_range:.2f})"
        )


@dataclass(frozen=True)
class ZonalAnisotropyModel:
    """Zonal anisotropy with different ranges in different directions.

    More general than geometric anisotropy - allows different sill values
    in different directions.

    Attributes:
        base_model: Base variogram model.
        x_range: Range in x-direction.
        y_range: Range in y-direction.
        z_range: Optional range in z-direction (for 3D).
    """

    base_model: VariogramModel
    x_range: float
    y_range: float
    z_range: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate zonal anisotropy."""
        if self.x_range <= 0 or self.y_range <= 0:
            raise ValueError("Ranges must be positive")
        if self.z_range is not None and self.z_range <= 0:
            raise ValueError("z_range must be positive if provided")

    def predict(
        self,
        distances: np.ndarray,
        direction: np.ndarray,
    ) -> np.ndarray:
        """Predict zonal anisotropic variogram.

        Args:
            distances: Array of distances.
            direction: Direction vectors (n_points, n_dims).

        Returns:
            Semi-variance values.
        """
        n_dims = direction.shape[1]
        ranges = [self.x_range, self.y_range]
        if n_dims == 3 and self.z_range is not None:
            ranges.append(self.z_range)

        # Compute effective range based on direction
        dir_norm = direction / (np.linalg.norm(direction, axis=1, keepdims=True) + 1e-10)
        effective_ranges = np.zeros(len(distances))

        for i, dir_vec in enumerate(dir_norm):
            # Weight ranges by direction components
            weights = np.abs(dir_vec[: len(ranges)])
            weights = weights / (np.sum(weights) + 1e-10)
            effective_ranges[i] = np.sum(weights * np.array(ranges))

        # Predict using effective ranges
        gamma = np.zeros(len(distances))
        for i, (dist, eff_range) in enumerate(zip(distances, effective_ranges)):
            temp_model = VariogramModel(
                model_type=self.base_model.model_type,
                nugget=self.base_model.nugget,
                sill=self.base_model.sill,
                range_param=eff_range,
                partial_sill=self.base_model.partial_sill,
                r_squared=self.base_model.r_squared,
            )
            gamma[i] = predict_variogram(temp_model, np.array([dist]))[0]

        return gamma

    def __repr__(self) -> str:
        """String representation."""
        z_str = f", z_range={self.z_range:.2f}" if self.z_range else ""
        return (
            f"ZonalAnisotropyModel(x_range={self.x_range:.2f}, "
            f"y_range={self.y_range:.2f}{z_str})"
        )


def compute_directional_variogram(
    points: PointSet,
    values: np.ndarray,
    direction: float,
    angle_tolerance: float = 22.5,
    n_lags: int = 15,
    max_lag: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute experimental variogram in a specific direction.

    Useful for detecting anisotropy by comparing variograms in different directions.

    Args:
        points: PointSet with sample locations.
        values: Sample values (n_samples,).
        direction: Direction angle in degrees (0 = east, counterclockwise).
        angle_tolerance: Tolerance for direction in degrees (default: 22.5°).
        n_lags: Number of lag bins.
        max_lag: Maximum lag distance.

    Returns:
        Tuple of (lags, semi_variances, n_pairs).
    """
    coordinates = points.coordinates
    if coordinates.shape[1] != 2:
        raise ValueError("Directional variograms currently only supported for 2D")

    n_samples = len(values)
    direction_rad = np.deg2rad(direction)
    tolerance_rad = np.deg2rad(angle_tolerance)

    # Direction vector
    dir_vec = np.array([np.cos(direction_rad), np.sin(direction_rad)])

    # Compute max distance if not provided
    if max_lag is None:
        distances_all = np.sqrt(
            np.sum((coordinates[:, None, :] - coordinates[None, :, :]) ** 2, axis=2)
        )
        max_lag = np.max(distances_all) / 2.0

    # Create lag bins
    lag_bins = np.linspace(0, max_lag, n_lags + 1)
    lag_centers = (lag_bins[:-1] + lag_bins[1:]) / 2.0

    semi_variance_sum = np.zeros(n_lags)
    n_pairs = np.zeros(n_lags, dtype=int)

    # Compute variogram only for pairs in specified direction
    for i in range(n_samples - 1):
        for j in range(i + 1, n_samples):
            # Vector from i to j
            vec = coordinates[j] - coordinates[i]
            dist = np.linalg.norm(vec)

            if dist == 0:
                continue

            # Normalize vector
            vec_norm = vec / dist

            # Check if direction matches (within tolerance)
            dot_product = np.dot(vec_norm, dir_vec)
            angle_diff = np.arccos(np.clip(dot_product, -1.0, 1.0))

            if angle_diff > tolerance_rad and (np.pi - angle_diff) > tolerance_rad:
                continue  # Skip if not in direction

            # Compute semi-variance
            semi_var = 0.5 * (values[i] - values[j]) ** 2

            # Assign to lag bin
            for k in range(n_lags):
                if lag_bins[k] <= dist < lag_bins[k + 1]:
                    semi_variance_sum[k] += semi_var
                    n_pairs[k] += 1
                    break

    # Average semi-variances
    semi_variances = np.where(n_pairs > 0, semi_variance_sum / n_pairs, np.nan)

    return lag_centers, semi_variances, n_pairs


def fit_nested_variogram(
    points: PointSet,
    values: np.ndarray,
    n_components: int = 2,
    component_types: Optional[list[str]] = None,
    n_lags: int = 15,
) -> NestedVariogramModel:
    """Fit nested variogram structure with multiple components.

    Fits multiple variogram models to capture different scales of variability.

    Args:
        points: PointSet with sample locations.
        values: Sample values (n_samples,).
        n_components: Number of nested components (default: 2).
        component_types: List of model types for each component.
            Defaults to ['spherical', 'spherical'].
        n_lags: Number of lag bins for experimental variogram.

    Returns:
        Fitted NestedVariogramModel.

    Example:
        >>> # Fit nested structure: nugget + short-range + long-range
        >>> nested = fit_nested_variogram(
        ...     points, values, n_components=3,
        ...     component_types=['spherical', 'spherical', 'exponential']
        ... )
    """
    if component_types is None:
        component_types = ["spherical"] * n_components

    if len(component_types) != n_components:
        raise ValueError(
            f"component_types length ({len(component_types)}) must match "
            f"n_components ({n_components})"
        )

    # Compute experimental variogram
    lags, semi_vars, n_pairs = compute_experimental_variogram(
        points, values, n_lags=n_lags
    )

    # Remove bins with insufficient pairs
    valid_mask = n_pairs > 0
    lags = lags[valid_mask]
    semi_vars = semi_vars[valid_mask]

    if len(lags) < 3:
        raise ValueError("Insufficient data for nested variogram fitting")

    # Fit components sequentially
    components = []
    remaining_variance = semi_vars.copy()

    for i, model_type in enumerate(component_types):
        if i == n_components - 1:
            # Last component gets all remaining variance
            component = fit_variogram_model(lags, remaining_variance, model_type=model_type)
        else:
            # Fit component to current variance
            component = fit_variogram_model(lags, remaining_variance, model_type=model_type)

            # Subtract fitted component
            fitted = predict_variogram(component, lags)
            remaining_variance = np.maximum(0, remaining_variance - fitted)

        components.append(component)

    return NestedVariogramModel(components=tuple(components))

