"""Unified geostatistical workflow interface.

Provides high-level interface for complete geostatistical workflows:
- Automatic variogram fitting
- Kriging estimation
- Cross-validation
- Uncertainty quantification
"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from geosmith.objects.pointset import PointSet
from geosmith.objects.rastergrid import RasterGrid
from geosmith.primitives.kriging import (
    IndicatorKriging,
    OrdinaryKriging,
    SimpleKriging,
    UniversalKriging,
)
from geosmith.primitives.kriging_cv import (
    CrossValidationResult,
    k_fold_cross_validation,
    leave_one_out_cross_validation,
)
from geosmith.primitives.simulation import sequential_gaussian_simulation
from geosmith.primitives.variogram import (
    VariogramModel,
    compute_experimental_variogram,
    fit_variogram_model,
)


@dataclass
class GeostatisticalResult:
    """Results from geostatistical estimation workflow.

    Attributes:
        estimates: Estimated values at target locations.
        variance: Estimation variance (uncertainty).
        cv_result: Cross-validation results (if validation performed).
        variogram_model: Fitted variogram model.
        realizations: Simulation realizations (if simulation performed).
    """

    estimates: np.ndarray
    variance: np.ndarray
    cv_result: Optional[CrossValidationResult] = None
    variogram_model: Optional[VariogramModel] = None
    realizations: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        """String representation."""
        cv_str = f", CV R²={self.cv_result.r2:.3f}" if self.cv_result else ""
        return (
            f"GeostatisticalResult(n_estimates={len(self.estimates)}, "
            f"mean={self.estimates.mean():.2f}{cv_str})"
        )


class GeostatisticalModel:
    """Unified interface for geostatistical modeling workflows.

    Handles the complete workflow from data to estimates:
    1. Variogram analysis and fitting
    2. Kriging estimation
    3. Cross-validation (optional)
    4. Uncertainty quantification (optional)

    Example:
        >>> from geosmith import PointSet
        >>> from geosmith.workflows.geostatistics import GeostatisticalModel
        >>>
        >>> # Create model
        >>> model = GeostatisticalModel(
        ...     data=pointset,
        ...     variable="grade",
        ...     method="kriging",
        ...     validation="cross_validate"
        ... )
        >>>
        >>> # Estimate on grid
        >>> results = model.estimate(grid_points)
        >>> print(f"RMSE: {results.cv_result.rmse:.2f}")
    """

    def __init__(
        self,
        data: PointSet,
        values: np.ndarray,
        method: Literal["kriging", "sgs", "ik"] = "kriging",
        kriging_type: Literal["ordinary", "simple", "universal"] = "ordinary",
        validation: Literal["none", "cross_validate", "k_fold"] = "none",
        n_folds: int = 5,
        variogram_model: Optional[VariogramModel] = None,
        variogram_model_type: str = "spherical",
        n_lags: int = 15,
        mean: Optional[float] = None,
        drift_terms: Optional[list[str]] = None,
        threshold: Optional[float] = None,
        n_realizations: int = 0,
        random_seed: Optional[int] = None,
    ):
        """Initialize geostatistical model.

        Args:
            data: PointSet with sample locations.
            values: Sample values (n_samples,).
            method: Estimation method ('kriging', 'sgs', 'ik').
            kriging_type: Type of kriging ('ordinary', 'simple', 'universal').
            validation: Validation method ('none', 'cross_validate', 'k_fold').
            n_folds: Number of folds for k-fold validation.
            variogram_model: Pre-fitted variogram model (optional).
            variogram_model_type: Variogram model type if fitting automatically.
            n_lags: Number of lag bins for variogram computation.
            mean: Known mean for Simple Kriging.
            drift_terms: Drift terms for Universal Kriging.
            threshold: Threshold for Indicator Kriging.
            n_realizations: Number of SGS realizations (0 = no simulation).
            random_seed: Random seed for reproducibility.
        """
        self.data = data
        self.values = values
        self.method = method
        self.kriging_type = kriging_type
        self.validation = validation
        self.n_folds = n_folds
        self.variogram_model = variogram_model
        self.variogram_model_type = variogram_model_type
        self.n_lags = n_lags
        self.mean = mean
        self.drift_terms = drift_terms
        self.threshold = threshold
        self.n_realizations = n_realizations
        self.random_seed = random_seed

        # Will be set during fit
        self._kriging_model = None
        self._cv_result = None

    def _fit_variogram(self) -> VariogramModel:
        """Fit variogram model to data."""
        if self.variogram_model is not None:
            return self.variogram_model

        # Compute experimental variogram
        lags, semi_vars, _ = compute_experimental_variogram(
            self.data, self.values, n_lags=self.n_lags
        )

        # Fit model
        variogram = fit_variogram_model(
            lags, semi_vars, model_type=self.variogram_model_type
        )

        return variogram

    def _fit_kriging(self) -> None:
        """Fit kriging model to data."""
        variogram = self._fit_variogram()

        if self.method == "ik":
            if self.threshold is None:
                # Use median as default threshold
                self.threshold = float(np.median(self.values))
            self._kriging_model = IndicatorKriging(
                variogram_model=variogram, threshold=self.threshold
            )
        elif self.kriging_type == "ordinary":
            self._kriging_model = OrdinaryKriging(variogram_model=variogram)
        elif self.kriging_type == "simple":
            if self.mean is None:
                self.mean = float(np.mean(self.values))
            self._kriging_model = SimpleKriging(
                variogram_model=variogram, mean=self.mean
            )
        elif self.kriging_type == "universal":
            self._kriging_model = UniversalKriging(
                variogram_model=variogram, drift_terms=self.drift_terms or ["linear"]
            )
        else:
            raise ValueError(f"Unknown kriging_type: {self.kriging_type}")

        self._kriging_model.fit(self.data, self.values)

    def _cross_validate(self) -> Optional[CrossValidationResult]:
        """Perform cross-validation."""
        if self.validation == "none":
            return None

        variogram = self._fit_variogram()

        if self.validation == "cross_validate":
            return leave_one_out_cross_validation(
                self.data,
                self.values,
                variogram,
                kriging_type=self.kriging_type,
                mean=self.mean,
                drift_terms=self.drift_terms,
            )
        elif self.validation == "k_fold":
            return k_fold_cross_validation(
                self.data,
                self.values,
                variogram,
                n_folds=self.n_folds,
                kriging_type=self.kriging_type,
                mean=self.mean,
                drift_terms=self.drift_terms,
            )
        else:
            return None

    def estimate(
        self,
        query_points: PointSet | RasterGrid,
        return_variance: bool = True,
    ) -> GeostatisticalResult:
        """Estimate values at target locations.

        Args:
            query_points: PointSet or RasterGrid with target locations.
            return_variance: Whether to return estimation variance.

        Returns:
            GeostatisticalResult with estimates, variance, and diagnostics.
        """
        # Convert RasterGrid to PointSet if needed
        if isinstance(query_points, RasterGrid):
            # Extract grid coordinates from affine transform
            grid = query_points
            if grid.data.ndim == 2:
                n_rows, n_cols = grid.data.shape
            else:
                _, n_rows, n_cols = grid.data.shape

            # Get transform parameters
            a, b, c, d, e, f = grid.transform

            # Generate grid coordinates
            col_indices = np.arange(n_cols)
            row_indices = np.arange(n_rows)
            col_coords, row_coords = np.meshgrid(col_indices, row_indices)

            # Apply affine transform: x = a*col + b*row + c, y = d*col + e*row + f
            x_coords = a * col_coords + b * row_coords + c
            y_coords = d * col_coords + e * row_coords + f

            # Flatten and stack
            coords = np.column_stack([x_coords.ravel(), y_coords.ravel()])
            query_points = PointSet(coordinates=coords)

        # Fit kriging model
        self._fit_kriging()

        if self._kriging_model is None:
            raise ValueError("Kriging model not fitted")

        # Predict
        result = self._kriging_model.predict(query_points, return_variance=return_variance)

        # Cross-validate if requested
        cv_result = None
        if self.validation != "none":
            cv_result = self._cross_validate()
            self._cv_result = cv_result

        # Generate realizations if requested
        realizations = None
        if self.n_realizations > 0 and self.method == "sgs":
            variogram = self._fit_variogram()
            realizations = sequential_gaussian_simulation(
                self.data,
                self.values,
                query_points,
                variogram,
                n_realizations=self.n_realizations,
                random_seed=self.random_seed,
            )

        return GeostatisticalResult(
            estimates=result.predictions,
            variance=result.variance,
            cv_result=cv_result,
            variogram_model=self._fit_variogram(),
            realizations=realizations,
        )

