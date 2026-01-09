"""Kriging primitives for spatial interpolation.

Pure kriging operations that work with Layer 1 objects.
Migrated from pygeomodeling.kriging.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist

from geosmith.objects.pointset import PointSet
from geosmith.primitives.base import BaseSpatialModel
from geosmith.primitives.variogram import VariogramModel, predict_variogram

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator if not args else decorator(args[0])


@dataclass
class KrigingResult:
    """Container for kriging predictions and diagnostics.

    Attributes:
        predictions: Predicted values at target locations.
        variance: Kriging variance (prediction uncertainty).
        weights: Optional kriging weights for each sample.
        lagrange_multiplier: Optional Lagrange multiplier from kriging system.
    """

    predictions: np.ndarray
    variance: np.ndarray
    weights: Optional[np.ndarray] = None
    lagrange_multiplier: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"KrigingResult(n_predictions={len(self.predictions)}, "
            f"mean_prediction={self.predictions.mean():.4f}, "
            f"mean_variance={self.variance.mean():.4f})"
        )


@njit(cache=True, fastmath=True)
def _compute_distances_fast(
    point: np.ndarray, coordinates: np.ndarray
) -> np.ndarray:
    """Numba-accelerated Euclidean distance computation.

    5-10x faster than scipy.spatial.distance.cdist for single point queries.
    """
    n_points = coordinates.shape[0]
    n_dims = coordinates.shape[1]
    distances = np.empty(n_points)

    for i in range(n_points):
        dist_sq = 0.0
        for d in range(n_dims):
            diff = point[d] - coordinates[i, d]
            dist_sq += diff * diff
        distances[i] = np.sqrt(dist_sq)

    return distances


class OrdinaryKriging(BaseSpatialModel):
    """Ordinary Kriging interpolation.

    Assumes a constant but unknown mean. Compatible with GeoSmith's BaseSpatialModel.

    Attributes:
        variogram_model: Fitted variogram model.
        regularization: Small value added to diagonal for stability.
    """

    def __init__(
        self,
        variogram_model: VariogramModel,
        regularization: float = 1e-10,
    ):
        """Initialize Ordinary Kriging.

        Args:
            variogram_model: Fitted variogram model.
            regularization: Small value added to diagonal for stability.
        """
        super().__init__()
        self.variogram_model = variogram_model
        self.regularization = regularization
        self.coordinates: Optional[np.ndarray] = None
        self.values: Optional[np.ndarray] = None
        self.K_inv: Optional[np.ndarray] = None

        # Set tags
        self.tags["supports_3d"] = True
        self.tags["supports_vector"] = True
        self.tags["requires_projected_crs"] = False

    def fit(
        self, points: PointSet, values: np.ndarray
    ) -> "OrdinaryKriging":
        """Fit kriging system to training data.

        Args:
            points: PointSet with sample locations.
            values: Sample values (n_samples,).

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If inputs are invalid.
        """
        coordinates = points.coordinates

        if len(coordinates) != len(values):
            raise ValueError(
                f"Coordinates ({len(coordinates)}) and values ({len(values)}) "
                f"must have same length"
            )

        if len(values) < 3:
            raise ValueError(
                f"Need at least 3 samples for kriging, got {len(values)}"
            )

        self.coordinates = coordinates
        self.values = values
        n = len(values)

        # Compute covariance matrix K
        # For a variogram γ(h), covariance C(h) = sill - γ(h)
        distances = cdist(coordinates, coordinates)
        gamma_matrix = predict_variogram(self.variogram_model, distances)
        K = self.variogram_model.sill - gamma_matrix

        # Add regularization to diagonal
        K += self.regularization * np.eye(n)

        # Build augmented kriging matrix with Lagrange multiplier
        K_aug = np.zeros((n + 1, n + 1))
        K_aug[:n, :n] = K
        K_aug[:n, n] = 1
        K_aug[n, :n] = 1
        K_aug[n, n] = 0

        # Invert once for efficiency
        try:
            self.K_inv = np.linalg.inv(K_aug)
        except np.linalg.LinAlgError:
            # Increase regularization if singular
            K += self.regularization * 100 * np.eye(n)
            K_aug[:n, :n] = K
            self.K_inv = np.linalg.inv(K_aug)

        self._fitted = True
        return self

    def predict(
        self,
        query_points: PointSet,
        return_variance: bool = True,
    ) -> KrigingResult:
        """Predict at target locations.

        Args:
            query_points: PointSet with target locations.
            return_variance: Whether to return kriging variance.

        Returns:
            KrigingResult with predictions and variance.

        Raises:
            ValueError: If model not fitted.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if self.K_inv is None:
            raise ValueError("Kriging system not initialized")

        coordinates_target = query_points.coordinates
        n_targets = coordinates_target.shape[0]
        n_samples = len(self.values)  # type: ignore

        predictions = np.zeros(n_targets)
        variances = np.zeros(n_targets) if return_variance else None

        # Predict each target point
        for i in range(n_targets):
            target = coordinates_target[i]

            # Compute covariance vector k between samples and target
            if NUMBA_AVAILABLE:
                distances = _compute_distances_fast(target, self.coordinates)  # type: ignore
            else:
                distances = cdist(
                    self.coordinates, target.reshape(1, -1)  # type: ignore
                ).ravel()

            gamma_vector = predict_variogram(self.variogram_model, distances)
            k = self.variogram_model.sill - gamma_vector

            # Augment with 1 for Lagrange multiplier
            k_aug = np.zeros(n_samples + 1)
            k_aug[:n_samples] = k
            k_aug[n_samples] = 1

            # Solve kriging system: weights = K_inv @ k_aug
            weights_aug = self.K_inv @ k_aug  # type: ignore
            weights = weights_aug[:n_samples]
            lagrange = weights_aug[n_samples]

            # Prediction: weighted sum
            predictions[i] = np.dot(weights, self.values)  # type: ignore

            # Kriging variance: C(0) - w'k - μ
            if return_variance:
                C_0 = (
                    self.variogram_model.sill - self.variogram_model.nugget
                )
                variances[i] = C_0 - np.dot(weights, k) - lagrange
                variances[i] = max(variances[i], 0.0)  # Ensure non-negative

        return KrigingResult(
            predictions=predictions,
            variance=variances if return_variance else np.zeros(n_targets),
            weights=None,  # Could store if needed
            lagrange_multiplier=None,
        )

