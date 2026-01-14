"""Cross-validation for geostatistical models.

Provides leave-one-out and k-fold cross-validation for kriging models
to assess prediction quality and validate variogram model choice.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from geosmith.objects.pointset import PointSet
from geosmith.primitives.kriging import (
    IndicatorKriging,
    OrdinaryKriging,
    SimpleKriging,
    UniversalKriging,
)
from geosmith.primitives.variogram import VariogramModel
from geosmith.utils.optional_imports import optional_import_single

SKLEARN_AVAILABLE, _ = optional_import_single("sklearn.model_selection", "KFold")
if SKLEARN_AVAILABLE:
    from sklearn.model_selection import KFold  # type: ignore
else:
    KFold = None  # type: ignore


@dataclass
class CrossValidationResult:
    """Results from cross-validation.

    Attributes:
        predictions: Cross-validated predictions (n_samples,).
        errors: Prediction errors (observed - predicted).
        mae: Mean Absolute Error.
        rmse: Root Mean Squared Error.
        r2: Coefficient of determination (R²).
        mean_error: Mean error (bias).
        std_error: Standard deviation of errors.
    """

    predictions: np.ndarray
    errors: np.ndarray
    mae: float
    rmse: float
    r2: float
    mean_error: float
    std_error: float

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CrossValidationResult(MAE={self.mae:.4f}, RMSE={self.rmse:.4f}, "
            f"R²={self.r2:.4f}, Bias={self.mean_error:.4f})"
        )


def leave_one_out_cross_validation(
    points: PointSet,
    values: np.ndarray,
    variogram_model: VariogramModel,
    kriging_type: Literal["ordinary", "simple", "universal"] = "ordinary",
    mean: float | None = None,
    drift_terms: list[str] | None = None,
    regularization: float = 1e-10,
) -> CrossValidationResult:
    """Perform leave-one-out cross-validation for kriging.

    For each sample point, fit kriging on all other points and predict at that point.
    This provides an unbiased estimate of prediction quality.

    Args:
        points: PointSet with sample locations.
        values: Sample values (n_samples,).
        variogram_model: Fitted variogram model.
        kriging_type: Type of kriging ('ordinary', 'simple', 'universal').
        mean: Known mean for Simple Kriging (required if kriging_type='simple').
        drift_terms: Drift terms for Universal Kriging (default: ['linear']).
        regularization: Regularization parameter.

    Returns:
        CrossValidationResult with metrics and predictions.

    Example:
        >>> from geosmith import PointSet
        >>> from geosmith.primitives.kriging_cv import leave_one_out_cross_validation
        >>> from geosmith.primitives.variogram import fit_variogram_model
        >>>
        >>> # Fit variogram
        >>> variogram = fit_variogram_model(lags, semi_vars)
        >>>
        >>> # Cross-validate
        >>> cv_result = leave_one_out_cross_validation(
        ...     points, values, variogram, kriging_type="ordinary"
        ... )
        >>> print(f"RMSE: {cv_result.rmse:.2f}, R²: {cv_result.r2:.3f}")
    """
    coordinates = points.coordinates
    n_samples = len(values)

    if n_samples < 4:
        raise ValueError(
            f"Need at least 4 samples for cross-validation, got {n_samples}"
        )

    predictions = np.zeros(n_samples)
    errors = np.zeros(n_samples)

    # Leave-one-out: predict each point using all others
    for i in range(n_samples):
        # Training set: all points except i
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[i] = False

        train_coords = coordinates[train_mask]
        train_values = values[train_mask]
        train_points = PointSet(coordinates=train_coords)

        # Test point
        test_coord = coordinates[i : i + 1]
        test_point = PointSet(coordinates=test_coord)

        # Fit and predict
        if kriging_type == "ordinary":
            kriging = OrdinaryKriging(
                variogram_model=variogram_model, regularization=regularization
            )
        elif kriging_type == "simple":
            if mean is None:
                raise ValueError("mean is required for Simple Kriging")
            kriging = SimpleKriging(
                variogram_model=variogram_model, mean=mean, regularization=regularization
            )
        elif kriging_type == "universal":
            kriging = UniversalKriging(
                variogram_model=variogram_model,
                drift_terms=drift_terms or ["linear"],
                regularization=regularization,
            )
        else:
            raise ValueError(
                f"Unknown kriging_type: {kriging_type}. "
                f"Must be one of: 'ordinary', 'simple', 'universal'"
            )

        kriging.fit(train_points, train_values)
        result = kriging.predict(test_point, return_variance=False)
        predictions[i] = result.predictions[0]

    # Compute errors
    errors = values - predictions

    # Compute metrics
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    mean_error = np.mean(errors)
    std_error = np.std(errors)

    # R²
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((values - np.mean(values)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return CrossValidationResult(
        predictions=predictions,
        errors=errors,
        mae=mae,
        rmse=rmse,
        r2=r2,
        mean_error=mean_error,
        std_error=std_error,
    )


def k_fold_cross_validation(
    points: PointSet,
    values: np.ndarray,
    variogram_model: VariogramModel,
    n_folds: int = 5,
    kriging_type: Literal["ordinary", "simple", "universal"] = "ordinary",
    mean: float | None = None,
    drift_terms: list[str] | None = None,
    regularization: float = 1e-10,
    random_state: int | None = None,
) -> CrossValidationResult:
    """Perform k-fold cross-validation for kriging.

    Splits data into k folds, fits on k-1 folds, predicts on remaining fold.
    More efficient than leave-one-out for large datasets.

    Args:
        points: PointSet with sample locations.
        values: Sample values (n_samples,).
        variogram_model: Fitted variogram model.
        n_folds: Number of folds (default: 5).
        kriging_type: Type of kriging ('ordinary', 'simple', 'universal').
        mean: Known mean for Simple Kriging.
        drift_terms: Drift terms for Universal Kriging.
        regularization: Regularization parameter.
        random_state: Random seed for fold splitting.

    Returns:
        CrossValidationResult with metrics and predictions.

    Raises:
        ImportError: If sklearn is not available.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "sklearn is required for k-fold cross-validation. "
            "Install with: pip install scikit-learn"
        )

    coordinates = points.coordinates
    n_samples = len(values)

    if n_samples < n_folds:
        raise ValueError(
            f"Need at least {n_folds} samples for {n_folds}-fold CV, got {n_samples}"
        )

    predictions = np.zeros(n_samples)
    errors = np.zeros(n_samples)

    # Create k-fold splitter
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)  # type: ignore

    # Cross-validate
    for train_idx, test_idx in kf.split(coordinates):
        # Training set
        train_coords = coordinates[train_idx]
        train_values = values[train_idx]
        train_points = PointSet(coordinates=train_coords)

        # Test set
        test_coords = coordinates[test_idx]
        test_points = PointSet(coordinates=test_coords)

        # Fit and predict
        if kriging_type == "ordinary":
            kriging = OrdinaryKriging(
                variogram_model=variogram_model, regularization=regularization
            )
        elif kriging_type == "simple":
            if mean is None:
                raise ValueError("mean is required for Simple Kriging")
            kriging = SimpleKriging(
                variogram_model=variogram_model, mean=mean, regularization=regularization
            )
        elif kriging_type == "universal":
            kriging = UniversalKriging(
                variogram_model=variogram_model,
                drift_terms=drift_terms or ["linear"],
                regularization=regularization,
            )
        else:
            raise ValueError(
                f"Unknown kriging_type: {kriging_type}. "
                f"Must be one of: 'ordinary', 'simple', 'universal'"
            )

        kriging.fit(train_points, train_values)
        result = kriging.predict(test_points, return_variance=False)
        predictions[test_idx] = result.predictions

    # Compute errors
    errors = values - predictions

    # Compute metrics
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    mean_error = np.mean(errors)
    std_error = np.std(errors)

    # R²
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((values - np.mean(values)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return CrossValidationResult(
        predictions=predictions,
        errors=errors,
        mae=mae,
        rmse=rmse,
        r2=r2,
        mean_error=mean_error,
        std_error=std_error,
    )

