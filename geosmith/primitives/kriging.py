"""Kriging primitives for spatial interpolation.

Pure kriging operations that work with Layer 1 objects.
Migrated from pygeomodeling.kriging.

Supports:
- Ordinary Kriging (OK): Constant but unknown mean
- Simple Kriging (SK): Known mean
- Universal Kriging (UK): Spatial trend modeling
- Indicator Kriging (IK): Categorical variables and risk assessment
- Co-Kriging (CK): Multiple correlated variables
"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from geosmith.objects.pointset import PointSet
from geosmith.primitives.base import BaseSpatialModel
from geosmith.primitives.variogram import VariogramModel, predict_variogram
from geosmith.utils.optional_imports import optional_import_single

# Optional dependencies
SCIPY_AVAILABLE, cdist_module = optional_import_single("scipy.spatial.distance", "cdist")
if SCIPY_AVAILABLE:
    from scipy.spatial.distance import cdist  # type: ignore
else:
    cdist = None  # type: ignore

NUMBA_AVAILABLE, njit_func = optional_import_single("numba", "njit")
if NUMBA_AVAILABLE:
    from numba import njit  # type: ignore
else:
    def njit(*args, **kwargs):  # type: ignore
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
def _compute_distances_fast(point: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    """Numba-accelerated Euclidean distance computation."""
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

    def fit(self, points: PointSet, values: np.ndarray) -> "OrdinaryKriging":
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
            raise ValueError(f"Need at least 3 samples for kriging, got {len(values)}")

        self.coordinates = coordinates
        self.values = values
        n = len(values)

        # Compute covariance matrix K
        # For a variogram γ(h), covariance C(h) = sill - γ(h)
        if not SCIPY_AVAILABLE:
            raise ImportError(
                "scipy is required for kriging. Install with: "
                "pip install geosmith[primitives] or pip install scipy"
            )
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
                if not SCIPY_AVAILABLE:
                    raise ImportError(
                        "scipy is required for kriging. Install with: "
                "pip install geosmith[primitives] or pip install scipy"
                    )
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
                C_0 = self.variogram_model.sill - self.variogram_model.nugget
                variances[i] = C_0 - np.dot(weights, k) - lagrange
                variances[i] = max(variances[i], 0.0)  # Ensure non-negative

        return KrigingResult(
            predictions=predictions,
            variance=variances if return_variance else np.zeros(n_targets),
            weights=None,  # Could store if needed
            lagrange_multiplier=None,
        )


class SimpleKriging(BaseSpatialModel):
    """Simple Kriging interpolation with known mean.

    Assumes a known constant mean. Simpler than Ordinary Kriging but requires
    prior knowledge of the mean value.

    Attributes:
        variogram_model: Fitted variogram model.
        mean: Known mean value.
        regularization: Small value added to diagonal for stability.
    """

    def __init__(
        self,
        variogram_model: VariogramModel,
        mean: float,
        regularization: float = 1e-10,
    ):
        """Initialize Simple Kriging.

        Args:
            variogram_model: Fitted variogram model.
            mean: Known mean value.
            regularization: Small value added to diagonal for stability.
        """
        super().__init__()
        self.variogram_model = variogram_model
        self.mean = mean
        self.regularization = regularization
        self.coordinates: Optional[np.ndarray] = None
        self.values: Optional[np.ndarray] = None
        self.K_inv: Optional[np.ndarray] = None

        # Set tags
        self.tags["supports_3d"] = True
        self.tags["supports_vector"] = True
        self.tags["requires_projected_crs"] = False

    def fit(self, points: PointSet, values: np.ndarray) -> "SimpleKriging":
        """Fit kriging system to training data.

        Args:
            points: PointSet with sample locations.
            values: Sample values (n_samples,).

        Returns:
            Self for method chaining.
        """
        coordinates = points.coordinates

        if len(coordinates) != len(values):
            raise ValueError(
                f"Coordinates ({len(coordinates)}) and values ({len(values)}) "
                f"must have same length"
            )

        if len(values) < 3:
            raise ValueError(f"Need at least 3 samples for kriging, got {len(values)}")

        self.coordinates = coordinates
        self.values = values
        n = len(values)

        # Center values around known mean
        self.values_centered = values - self.mean

        # Compute covariance matrix K
        if not SCIPY_AVAILABLE:
            raise ImportError(
                "scipy is required for kriging. Install with: "
                "pip install geosmith[primitives] or pip install scipy"
            )
        distances = cdist(coordinates, coordinates)
        gamma_matrix = predict_variogram(self.variogram_model, distances)
        K = self.variogram_model.sill - gamma_matrix

        # Add regularization
        K += self.regularization * np.eye(n)

        # Invert (no Lagrange multiplier needed for Simple Kriging)
        try:
            self.K_inv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            K += self.regularization * 100 * np.eye(n)
            self.K_inv = np.linalg.inv(K)

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
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if self.K_inv is None:
            raise ValueError("Kriging system not initialized")

        coordinates_target = query_points.coordinates
        n_targets = coordinates_target.shape[0]
        n_samples = len(self.values_centered)  # type: ignore

        predictions = np.zeros(n_targets)
        variances = np.zeros(n_targets) if return_variance else None

        for i in range(n_targets):
            target = coordinates_target[i]

            # Compute covariance vector k
            if NUMBA_AVAILABLE:
                distances = _compute_distances_fast(target, self.coordinates)  # type: ignore
            else:
                distances = cdist(
                    self.coordinates, target.reshape(1, -1)  # type: ignore
                ).ravel()

            gamma_vector = predict_variogram(self.variogram_model, distances)
            k = self.variogram_model.sill - gamma_vector

            # Solve: weights = K_inv @ k
            weights = self.K_inv @ k  # type: ignore

            # Prediction: weighted sum of centered values + mean
            predictions[i] = np.dot(weights, self.values_centered) + self.mean  # type: ignore

            # Variance: C(0) - w'k
            if return_variance:
                C_0 = self.variogram_model.sill - self.variogram_model.nugget
                variances[i] = max(C_0 - np.dot(weights, k), 0.0)

        return KrigingResult(
            predictions=predictions,
            variance=variances if return_variance else np.zeros(n_targets),
        )


class UniversalKriging(BaseSpatialModel):
    """Universal Kriging with spatial trend modeling.

    Accounts for spatial trends by including drift terms (e.g., linear, quadratic).
    Useful when data exhibits non-stationary behavior.

    Attributes:
        variogram_model: Fitted variogram model.
        drift_terms: List of drift terms ('constant', 'linear', 'quadratic').
        regularization: Small value added to diagonal for stability.
    """

    def __init__(
        self,
        variogram_model: VariogramModel,
        drift_terms: list[Literal["constant", "linear", "quadratic"]] | None = None,
        regularization: float = 1e-10,
    ):
        """Initialize Universal Kriging.

        Args:
            variogram_model: Fitted variogram model.
            drift_terms: List of drift terms. Defaults to ['linear'].
            regularization: Small value added to diagonal for stability.
        """
        super().__init__()
        self.variogram_model = variogram_model
        self.drift_terms = drift_terms or ["linear"]
        self.regularization = regularization
        self.coordinates: Optional[np.ndarray] = None
        self.values: Optional[np.ndarray] = None
        self.K_inv: Optional[np.ndarray] = None
        self.n_drift: int = 0

        # Set tags
        self.tags["supports_3d"] = True
        self.tags["supports_vector"] = True
        self.tags["requires_projected_crs"] = False

    def _compute_drift_matrix(self, coordinates: np.ndarray) -> np.ndarray:
        """Compute drift matrix F from coordinates.

        Args:
            coordinates: Sample coordinates (n_samples, n_dims).

        Returns:
            Drift matrix F (n_samples, n_drift).
        """
        n = coordinates.shape[0]
        n_dims = coordinates.shape[1]
        drift_functions = []

        if "constant" in self.drift_terms:
            drift_functions.append(np.ones(n))

        if "linear" in self.drift_terms:
            for d in range(n_dims):
                drift_functions.append(coordinates[:, d])

        if "quadratic" in self.drift_terms:
            for d in range(n_dims):
                drift_functions.append(coordinates[:, d] ** 2)
            # Cross terms
            if n_dims >= 2:
                for i in range(n_dims):
                    for j in range(i + 1, n_dims):
                        drift_functions.append(coordinates[:, i] * coordinates[:, j])

        if not drift_functions:
            raise ValueError(
                f"No valid drift terms. Must be one of: {['constant', 'linear', 'quadratic']}"
            )

        F = np.column_stack(drift_functions)
        self.n_drift = F.shape[1]
        return F

    def fit(self, points: PointSet, values: np.ndarray) -> "UniversalKriging":
        """Fit universal kriging system to training data.

        Args:
            points: PointSet with sample locations.
            values: Sample values (n_samples,).

        Returns:
            Self for method chaining.
        """
        coordinates = points.coordinates

        if len(coordinates) != len(values):
            raise ValueError(
                f"Coordinates ({len(coordinates)}) and values ({len(values)}) "
                f"must have same length"
            )

        if len(values) < 3:
            raise ValueError(f"Need at least 3 samples for kriging, got {len(values)}")

        self.coordinates = coordinates
        self.values = values
        n = len(values)

        # Compute covariance matrix
        if not SCIPY_AVAILABLE:
            raise ImportError(
                "scipy is required for kriging. Install with: "
                "pip install geosmith[primitives] or pip install scipy"
            )
        distances = cdist(coordinates, coordinates)
        gamma_matrix = predict_variogram(self.variogram_model, distances)
        K = self.variogram_model.sill - gamma_matrix
        K += self.regularization * np.eye(n)

        # Compute drift matrix
        F = self._compute_drift_matrix(coordinates)
        m = F.shape[1]

        # Build augmented system: [K  F] [w] = [k]
        #                          [F' 0] [ν]   [f]
        K_aug = np.zeros((n + m, n + m))
        K_aug[:n, :n] = K
        K_aug[:n, n:] = F
        K_aug[n:, :n] = F.T

        # Invert
        try:
            self.K_inv = np.linalg.inv(K_aug)
        except np.linalg.LinAlgError:
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
        """Predict at target locations with trend.

        Args:
            query_points: PointSet with target locations.
            return_variance: Whether to return kriging variance.

        Returns:
            KrigingResult with predictions and variance.
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

        # Compute drift for targets
        F_target = self._compute_drift_matrix(coordinates_target)

        for i in range(n_targets):
            target = coordinates_target[i]

            # Covariance vector
            if NUMBA_AVAILABLE:
                distances = _compute_distances_fast(target, self.coordinates)  # type: ignore
            else:
                distances = cdist(
                    self.coordinates, target.reshape(1, -1)  # type: ignore
                ).ravel()

            gamma_vector = predict_variogram(self.variogram_model, distances)
            k = self.variogram_model.sill - gamma_vector

            # Augment with drift
            k_aug = np.zeros(n_samples + self.n_drift)
            k_aug[:n_samples] = k
            k_aug[n_samples:] = F_target[i]

            # Solve
            weights_aug = self.K_inv @ k_aug  # type: ignore
            weights = weights_aug[:n_samples]

            # Prediction
            predictions[i] = np.dot(weights, self.values)  # type: ignore

            # Variance
            if return_variance:
                C_0 = self.variogram_model.sill - self.variogram_model.nugget
                variances[i] = max(C_0 - np.dot(weights, k), 0.0)

        return KrigingResult(
            predictions=predictions,
            variance=variances if return_variance else np.zeros(n_targets),
        )


class IndicatorKriging(BaseSpatialModel):
    """Indicator Kriging for categorical variables and risk assessment.

    Transforms continuous variables into indicators (0/1) based on thresholds,
    then performs kriging on the indicator. Useful for:
    - Categorical variables (lithology, facies)
    - Risk assessment (probability of exceeding threshold)
    - Uncertainty quantification

    Attributes:
        variogram_model: Fitted variogram model for indicator.
        threshold: Threshold value for indicator transformation.
        regularization: Small value added to diagonal for stability.
    """

    def __init__(
        self,
        variogram_model: VariogramModel,
        threshold: float,
        regularization: float = 1e-10,
    ):
        """Initialize Indicator Kriging.

        Args:
            variogram_model: Fitted variogram model for indicator variable.
            threshold: Threshold value for indicator transformation.
            regularization: Small value added to diagonal for stability.
        """
        super().__init__()
        self.variogram_model = variogram_model
        self.threshold = threshold
        self.regularization = regularization
        self.coordinates: Optional[np.ndarray] = None
        self.values: Optional[np.ndarray] = None
        self.indicator_values: Optional[np.ndarray] = None
        self.K_inv: Optional[np.ndarray] = None

        # Set tags
        self.tags["supports_3d"] = True
        self.tags["supports_vector"] = True
        self.tags["requires_projected_crs"] = False

    def fit(self, points: PointSet, values: np.ndarray) -> "IndicatorKriging":
        """Fit indicator kriging system to training data.

        Args:
            points: PointSet with sample locations.
            values: Sample values (n_samples,). Will be transformed to indicators.

        Returns:
            Self for method chaining.
        """
        coordinates = points.coordinates

        if len(coordinates) != len(values):
            raise ValueError(
                f"Coordinates ({len(coordinates)}) and values ({len(values)}) "
                f"must have same length"
            )

        if len(values) < 3:
            raise ValueError(f"Need at least 3 samples for kriging, got {len(values)}")

        self.coordinates = coordinates
        self.values = values

        # Transform to indicator: I(z) = 1 if z >= threshold, else 0
        self.indicator_values = (values >= self.threshold).astype(float)
        n = len(self.indicator_values)

        # Compute covariance matrix for indicator
        if not SCIPY_AVAILABLE:
            raise ImportError(
                "scipy is required for kriging. Install with: "
                "pip install geosmith[primitives] or pip install scipy"
            )
        distances = cdist(coordinates, coordinates)
        gamma_matrix = predict_variogram(self.variogram_model, distances)
        K = self.variogram_model.sill - gamma_matrix

        # Add regularization
        K += self.regularization * np.eye(n)

        # Build augmented kriging matrix with Lagrange multiplier
        K_aug = np.zeros((n + 1, n + 1))
        K_aug[:n, :n] = K
        K_aug[:n, n] = 1
        K_aug[n, :n] = 1
        K_aug[n, n] = 0

        # Invert
        try:
            self.K_inv = np.linalg.inv(K_aug)
        except np.linalg.LinAlgError:
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
        """Predict indicator (probability) at target locations.

        Args:
            query_points: PointSet with target locations.
            return_variance: Whether to return kriging variance.

        Returns:
            KrigingResult with predictions (probabilities) and variance.
            Predictions are probabilities in [0, 1] that value >= threshold.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if self.K_inv is None:
            raise ValueError("Kriging system not initialized")

        coordinates_target = query_points.coordinates
        n_targets = coordinates_target.shape[0]
        n_samples = len(self.indicator_values)  # type: ignore

        predictions = np.zeros(n_targets)
        variances = np.zeros(n_targets) if return_variance else None

        for i in range(n_targets):
            target = coordinates_target[i]

            # Compute covariance vector k
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

            # Solve kriging system
            weights_aug = self.K_inv @ k_aug  # type: ignore
            weights = weights_aug[:n_samples]
            lagrange = weights_aug[n_samples]

            # Prediction: probability that value >= threshold
            predictions[i] = np.dot(weights, self.indicator_values)  # type: ignore
            # Clamp to [0, 1]
            predictions[i] = max(0.0, min(1.0, predictions[i]))

            # Variance
            if return_variance:
                C_0 = self.variogram_model.sill - self.variogram_model.nugget
                variances[i] = max(C_0 - np.dot(weights, k) - lagrange, 0.0)

        return KrigingResult(
            predictions=predictions,
            variance=variances if return_variance else np.zeros(n_targets),
        )


class CoKriging(BaseSpatialModel):
    """Co-Kriging for multiple correlated variables.

    Leverages cross-correlation between primary and secondary variables
    to improve predictions. Useful when secondary variable is densely
    sampled but primary is sparse.

    Attributes:
        primary_variogram: Variogram model for primary variable.
        secondary_variogram: Variogram model for secondary variable.
        cross_variogram: Cross-variogram model between variables.
        regularization: Small value added to diagonal for stability.
    """

    def __init__(
        self,
        primary_variogram: VariogramModel,
        secondary_variogram: VariogramModel,
        cross_variogram: VariogramModel,
        regularization: float = 1e-10,
    ):
        """Initialize Co-Kriging.

        Args:
            primary_variogram: Variogram model for primary variable.
            secondary_variogram: Variogram model for secondary variable.
            cross_variogram: Cross-variogram model between variables.
            regularization: Small value added to diagonal for stability.
        """
        super().__init__()
        self.primary_variogram = primary_variogram
        self.secondary_variogram = secondary_variogram
        self.cross_variogram = cross_variogram
        self.regularization = regularization
        self.primary_coords: Optional[np.ndarray] = None
        self.primary_values: Optional[np.ndarray] = None
        self.secondary_coords: Optional[np.ndarray] = None
        self.secondary_values: Optional[np.ndarray] = None
        self.K_inv: Optional[np.ndarray] = None

        # Set tags
        self.tags["supports_3d"] = True
        self.tags["supports_vector"] = True
        self.tags["requires_projected_crs"] = False

    def fit(
        self,
        primary_points: PointSet,
        primary_values: np.ndarray,
        secondary_points: PointSet,
        secondary_values: np.ndarray,
    ) -> "CoKriging":
        """Fit co-kriging system to training data.

        Args:
            primary_points: PointSet with primary variable sample locations.
            primary_values: Primary variable values (n_samples_primary,).
            secondary_points: PointSet with secondary variable sample locations.
            secondary_values: Secondary variable values (n_samples_secondary,).

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If inputs are invalid.
        """
        primary_coords = primary_points.coordinates
        secondary_coords = secondary_points.coordinates

        if len(primary_coords) != len(primary_values):
            raise ValueError(
                f"Primary coordinates ({len(primary_coords)}) and values "
                f"({len(primary_values)}) must have same length"
            )

        if len(secondary_coords) != len(secondary_values):
            raise ValueError(
                f"Secondary coordinates ({len(secondary_coords)}) and values "
                f"({len(secondary_values)}) must have same length"
            )

        if len(primary_values) < 3:
            raise ValueError(
                f"Need at least 3 primary samples, got {len(primary_values)}"
            )

        if len(secondary_values) < 3:
            raise ValueError(
                f"Need at least 3 secondary samples, got {len(secondary_values)}"
            )

        self.primary_coords = primary_coords
        self.primary_values = primary_values
        self.secondary_coords = secondary_coords
        self.secondary_values = secondary_values

        n1 = len(primary_values)
        n2 = len(secondary_values)
        n = n1 + n2

        # Build covariance matrix
        # [K11  K12]
        # [K21  K22]
        if not SCIPY_AVAILABLE:
            raise ImportError(
                "scipy is required for co-kriging. Install with: "
                "pip install geosmith[primitives] or pip install scipy"
            )

        K = np.zeros((n, n))

        # K11: primary-primary covariance
        dist11 = cdist(primary_coords, primary_coords)
        gamma11 = predict_variogram(self.primary_variogram, dist11)
        K[:n1, :n1] = self.primary_variogram.sill - gamma11

        # K22: secondary-secondary covariance
        dist22 = cdist(secondary_coords, secondary_coords)
        gamma22 = predict_variogram(self.secondary_variogram, dist22)
        K[n1:, n1:] = self.secondary_variogram.sill - gamma22

        # K12, K21: cross-covariance
        dist12 = cdist(primary_coords, secondary_coords)
        gamma12 = predict_variogram(self.cross_variogram, dist12)
        cross_cov = self.cross_variogram.sill - gamma12
        K[:n1, n1:] = cross_cov
        K[n1:, :n1] = cross_cov.T

        # Regularization
        K += self.regularization * np.eye(n)

        # Augment with Lagrange multipliers (2 constraints: one for each variable)
        K_aug = np.zeros((n + 2, n + 2))
        K_aug[:n, :n] = K
        K_aug[:n1, n] = 1  # Primary constraint
        K_aug[n, :n1] = 1
        K_aug[n1:n, n + 1] = 1  # Secondary constraint
        K_aug[n + 1, n1:n] = 1

        # Invert
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
        """Predict primary variable at target locations using both variables.

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
            raise ValueError("Co-kriging system not initialized")

        coordinates_target = query_points.coordinates
        n_targets = coordinates_target.shape[0]
        n1 = len(self.primary_values)  # type: ignore
        n2 = len(self.secondary_values)  # type: ignore
        n = n1 + n2

        predictions = np.zeros(n_targets)
        variances = np.zeros(n_targets) if return_variance else None

        for i in range(n_targets):
            target = coordinates_target[i]

            # Covariance vectors
            if NUMBA_AVAILABLE:
                dist_p = _compute_distances_fast(
                    target, self.primary_coords  # type: ignore
                )
            else:
                dist_p = cdist(
                    self.primary_coords, target.reshape(1, -1)  # type: ignore
                ).ravel()

            gamma_p = predict_variogram(self.primary_variogram, dist_p)
            k_p = self.primary_variogram.sill - gamma_p

            if NUMBA_AVAILABLE:
                dist_s = _compute_distances_fast(
                    target, self.secondary_coords  # type: ignore
                )
            else:
                dist_s = cdist(
                    self.secondary_coords, target.reshape(1, -1)  # type: ignore
                ).ravel()

            gamma_s = predict_variogram(self.cross_variogram, dist_s)
            k_s = self.cross_variogram.sill - gamma_s

            # Augment with constraints
            k_aug = np.zeros(n + 2)
            k_aug[:n1] = k_p
            k_aug[n1:n] = k_s
            k_aug[n] = 1  # Primary constraint
            k_aug[n + 1] = 0  # Secondary constraint (not used for prediction)

            # Solve
            weights_aug = self.K_inv @ k_aug  # type: ignore
            weights_p = weights_aug[:n1]
            weights_s = weights_aug[n1:n]

            # Prediction: weighted sum of both variables
            pred_p = np.dot(weights_p, self.primary_values)  # type: ignore
            pred_s = np.dot(weights_s, self.secondary_values)  # type: ignore
            predictions[i] = pred_p + pred_s

            # Variance
            if return_variance:
                C_0 = (
                    self.primary_variogram.sill
                    - self.primary_variogram.nugget
                )
                variances[i] = max(
                    C_0
                    - np.dot(weights_p, k_p)
                    - np.dot(weights_s, k_s)
                    - weights_aug[n],  # Lagrange multiplier
                    0.0,
                )

        return KrigingResult(
            predictions=predictions,
            variance=variances if return_variance else np.zeros(n_targets),
        )
