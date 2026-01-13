"""Compatibility shims for pygeomodeling.

Provides drop-in replacements for pygeomodeling classes and functions,
wrapping GeoSmith's new 4-layer architecture implementations.

This allows existing pygeomodeling code to work with minimal changes
while migrating to GeoSmith.
"""

import warnings
from typing import Optional

import numpy as np

from geosmith.objects.pointset import PointSet
from geosmith.primitives.kriging import (
    KrigingResult,
    OrdinaryKriging as GSOrdinaryKriging,
)
from geosmith.primitives.variogram import (
    VariogramModel as GSVariogramModel,
    compute_experimental_variogram as gs_compute_variogram,
    fit_variogram_model as gs_fit_variogram,
)
from geosmith.workflows.grdecl import read_grdecl


# Deprecation warning
_DEPRECATION_WARNING = (
    "pygeomodeling is deprecated. Use GeoSmith instead. "
    "See https://github.com/kylejones200/geosmith for migration guide."
)


class GRDECLParser:
    """Compatibility shim for pygeomodeling.GRDECLParser.

    Wraps GeoSmith's read_grdecl workflow function.
    """

    def __init__(self, filepath: str):
        """Initialize GRDECL parser.

        Args:
            filepath: Path to GRDECL file.
        """
        warnings.warn(_DEPRECATION_WARNING, DeprecationWarning, stacklevel=2)
        self.filepath = filepath
        self.grid_dimensions = None
        self.properties = {}

    def load_data(self) -> dict:
        """Load and parse GRDECL file.

        Returns:
            Dictionary with 'dimensions' and 'properties' keys.
        """
        data = read_grdecl(self.filepath)
        self.grid_dimensions = data["dimensions"]
        self.properties = data["properties"]
        return data

    def parse(self) -> dict:
        """Alias for load_data() for backward compatibility."""
        return self.load_data()

    def get_property_3d(self, property_name: str) -> Optional[np.ndarray]:
        """Get a 3D property array."""
        return self.properties.get(property_name)


class OrdinaryKriging:
    """Compatibility shim for pygeomodeling.kriging.OrdinaryKriging.

    Wraps GeoSmith's OrdinaryKriging primitive.
    """

    def __init__(
        self,
        variogram_model: GSVariogramModel,
        regularization: float = 1e-10,
    ):
        """Initialize Ordinary Kriging.

        Args:
            variogram_model: Fitted variogram model.
            regularization: Small value added to diagonal for stability.
        """
        warnings.warn(_DEPRECATION_WARNING, DeprecationWarning, stacklevel=2)
        self._gs_kriging = GSOrdinaryKriging(
            variogram_model=variogram_model, regularization=regularization
        )
        self.variogram_model = variogram_model
        self.regularization = regularization
        self.coordinates = None
        self.values = None

    def fit(self, coordinates: np.ndarray, values: np.ndarray):
        """Fit kriging system to training data.

        Args:
            coordinates: Sample coordinates (n_samples, n_dims).
            values: Sample values (n_samples,).

        Returns:
            Self for method chaining.
        """
        # Convert to PointSet
        points = PointSet(coordinates=coordinates)
        self._gs_kriging.fit(points, values)

        # Store for compatibility
        self.coordinates = coordinates
        self.values = values

        return self

    def predict(
        self,
        coordinates_target: np.ndarray,
        return_variance: bool = True,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict at target locations.

        Args:
            coordinates_target: Target coordinates (n_targets, n_dims).
            return_variance: Whether to return kriging variance.

        Returns:
            Tuple of (predictions, variance).
        """
        # Convert to PointSet
        query_points = PointSet(coordinates=coordinates_target)

        # Predict using GeoSmith
        result = self._gs_kriging.predict(query_points, return_variance=return_variance)

        if return_variance:
            return result.predictions, result.variance
        return result.predictions, None


class VariogramModel:
    """Compatibility shim for pygeomodeling.variogram.VariogramModel.

    Wraps GeoSmith's VariogramModel.
    """

    def __init__(
        self,
        model_type: str,
        nugget: float,
        sill: float,
        range_param: float,
        partial_sill: Optional[float] = None,
        r_squared: float = 0.0,
    ):
        """Initialize VariogramModel.

        Args:
            model_type: Type of model.
            nugget: Nugget effect.
            sill: Total sill.
            range_param: Range parameter.
            partial_sill: Partial sill (computed if not provided).
            r_squared: Goodness of fit.
        """
        warnings.warn(_DEPRECATION_WARNING, DeprecationWarning, stacklevel=2)
        if partial_sill is None:
            partial_sill = sill - nugget

        self._gs_model = GSVariogramModel(
            model_type=model_type,
            nugget=nugget,
            sill=sill,
            range_param=range_param,
            partial_sill=partial_sill,
            r_squared=r_squared,
        )

        # Expose attributes for compatibility
        self.model_type = model_type
        self.nugget = nugget
        self.sill = sill
        self.range_param = range_param
        self.partial_sill = partial_sill
        self.r_squared = r_squared

    def __str__(self) -> str:
        """String representation."""
        return str(self._gs_model)


def compute_experimental_variogram(
    coordinates: np.ndarray,
    values: np.ndarray,
    n_lags: int = 15,
    max_lag: Optional[float] = None,
    lag_tolerance: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility shim for pygeomodeling.variogram.compute_experimental_variogram.

    Args:
        coordinates: Sample coordinates (n_samples, n_dims).
        values: Sample values (n_samples,).
        n_lags: Number of lag bins.
        max_lag: Maximum lag distance.
        lag_tolerance: Tolerance for binning.

    Returns:
        Tuple of (lags, semi_variance, n_pairs).
    """
    warnings.warn(_DEPRECATION_WARNING, DeprecationWarning, stacklevel=2)

    # Convert to PointSet
    points = PointSet(coordinates=coordinates)

    # Use GeoSmith implementation
    return gs_compute_variogram(
        points, values, n_lags=n_lags, max_lag=max_lag, lag_tolerance=lag_tolerance
    )


def fit_variogram_model(
    lags: np.ndarray,
    semi_variances: np.ndarray,
    model_type: str = "spherical",
    initial_params: Optional[dict] = None,
) -> VariogramModel:
    """Compatibility shim for pygeomodeling.variogram.fit_variogram_model.

    Args:
        lags: Lag distances.
        semi_variances: Experimental semi-variances.
        model_type: Model type.
        initial_params: Optional initial parameter guesses.

    Returns:
        Fitted VariogramModel.
    """
    warnings.warn(_DEPRECATION_WARNING, DeprecationWarning, stacklevel=2)

    # Use GeoSmith implementation
    gs_model = gs_fit_variogram_model(
        lags, semi_variances, model_type=model_type, initial_params=initial_params
    )

    # Wrap in compatibility shim
    return VariogramModel(
        model_type=gs_model.model_type,
        nugget=gs_model.nugget,
        sill=gs_model.sill,
        range_param=gs_model.range_param,
        partial_sill=gs_model.partial_sill,
        r_squared=gs_model.r_squared,
    )


class UnifiedSPE9Toolkit:
    """Compatibility shim for pygeomodeling.unified_toolkit.UnifiedSPE9Toolkit.

    This is a placeholder - full migration would require more complex refactoring.
    For now, provides a deprecation warning and basic structure.
    """

    def __init__(self, data_path: Optional[str] = None, backend: str = "sklearn"):
        """Initialize UnifiedSPE9Toolkit.

        Args:
            data_path: Path to SPE9 dataset file.
            backend: Modeling backend ('sklearn' or 'gpytorch').
        """
        warnings.warn(
            f"{_DEPRECATION_WARNING} "
            "UnifiedSPE9Toolkit migration is in progress. "
            "Use GeoSmith's task-based workflows instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.data_path = data_path
        self.backend = backend
        # Placeholder - full implementation would require more work
        raise NotImplementedError(
            "UnifiedSPE9Toolkit migration to GeoSmith is in progress. "
            "Use GeoSmith's task-based workflows for now."
        )


# Convenience function
def load_spe9_data(filepath: str) -> dict:
    """Compatibility shim for pygeomodeling.grdecl_parser.load_spe9_data.

    Args:
        filepath: Path to SPE9 GRDECL file.

    Returns:
        Dictionary with 'dimensions' and 'properties' keys.
    """
    warnings.warn(_DEPRECATION_WARNING, DeprecationWarning, stacklevel=2)
    return read_grdecl(filepath)
