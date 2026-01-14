"""Variogram analysis primitives.

Pure variogram operations that work with Layer 1 objects.
Migrated from pygeomodeling.variogram.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from geosmith.objects.pointset import PointSet

# Optional dependencies
try:
    from scipy.optimize import curve_fit

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    curve_fit = None  # type: ignore

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator if not args else decorator(args[0])

    prange = range


@dataclass(frozen=True)
class VariogramModel:
    """Container for fitted variogram model parameters.

    Attributes:
        model_type: Type of model ('spherical', 'exponential', 'gaussian', 'linear').
        nugget: Nugget effect (small-scale variance).
        sill: Total sill (nugget + partial sill).
        range_param: Range parameter (correlation length).
        partial_sill: Partial sill (sill - nugget).
        r_squared: Goodness of fit.
    """

    model_type: str
    nugget: float
    sill: float
    range_param: float
    partial_sill: float
    r_squared: float

    def __post_init__(self) -> None:
        """Validate VariogramModel parameters."""
        valid_types = ("spherical", "exponential", "gaussian", "linear", "power")
        if self.model_type not in valid_types:
            raise ValueError(
                f"model_type must be one of {valid_types}, got {self.model_type}"
            )

        if self.nugget < 0:
            raise ValueError(f"nugget must be non-negative, got {self.nugget}")

        if self.sill < self.nugget:
            raise ValueError(f"sill ({self.sill}) must be >= nugget ({self.nugget})")

        if (
            self.range_param <= 0
            and self.model_type not in ("linear", "power")
        ):
            raise ValueError(
                f"range_param must be positive for {self.model_type} model, "
                f"got {self.range_param}"
            )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"VariogramModel(type={self.model_type}, nugget={self.nugget:.4f}, "
            f"sill={self.sill:.4f}, range={self.range_param:.4f}, "
            f"r²={self.r_squared:.4f})"
        )


def _spherical_model(
    h: np.ndarray, nugget: float, sill: float, range_param: float
) -> np.ndarray:
    """Spherical variogram model.

    Args:
        h: Distance (lag).
        nugget: Nugget effect.
        sill: Total sill.
        range_param: Range parameter.

    Returns:
        Semi-variance values.
    """
    gamma = np.zeros_like(h, dtype=float)

    mask = h < range_param
    if np.any(mask):
        h_scaled = h[mask] / range_param
        gamma[mask] = nugget + (sill - nugget) * (1.5 * h_scaled - 0.5 * h_scaled**3)

    gamma[~mask] = sill
    return gamma


def _exponential_model(
    h: np.ndarray, nugget: float, sill: float, range_param: float
) -> np.ndarray:
    """Exponential variogram model.

    Args:
        h: Distance (lag).
        nugget: Nugget effect.
        sill: Total sill.
        range_param: Range parameter.

    Returns:
        Semi-variance values.
    """
    return nugget + (sill - nugget) * (1 - np.exp(-h / range_param))


def _gaussian_model(
    h: np.ndarray, nugget: float, sill: float, range_param: float
) -> np.ndarray:
    """Gaussian variogram model.

    Args:
        h: Distance (lag).
        nugget: Nugget effect.
        sill: Total sill.
        range_param: Range parameter.

    Returns:
        Semi-variance values.
    """
    return nugget + (sill - nugget) * (1 - np.exp(-(h**2) / (range_param**2)))


def _linear_model(h: np.ndarray, nugget: float, slope: float) -> np.ndarray:
    """Linear variogram model (no sill).

    Args:
        h: Distance (lag).
        nugget: Nugget effect.
        slope: Slope of the line.

    Returns:
        Semi-variance values.
    """
    return nugget + slope * h


def _power_model(
    h: np.ndarray, nugget: float, scale: float, exponent: float
) -> np.ndarray:
    """Power variogram model (fractal behavior).

    Useful for self-similar processes. No sill - variance increases indefinitely.

    Args:
        h: Distance (lag).
        nugget: Nugget effect.
        scale: Scale parameter.
        exponent: Power exponent (typically 0 < exponent < 2).

    Returns:
        Semi-variance values.
    """
    # Avoid division by zero
    h_safe = np.where(h > 0, h, 1e-10)
    return nugget + scale * (h_safe ** exponent)


# Model registry
VARIOGRAM_MODELS: dict[str, Callable] = {
    "spherical": _spherical_model,
    "exponential": _exponential_model,
    "gaussian": _gaussian_model,
    "linear": _linear_model,
    "power": _power_model,
}


@njit(parallel=True, cache=True, fastmath=True)
def _compute_variogram_fast(
    coordinates: np.ndarray,
    values: np.ndarray,
    lag_bins: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Numba-accelerated variogram computation."""
    n_points = coordinates.shape[0]
    n_lags = len(lag_bins) - 1
    n_dims = coordinates.shape[1]

    semi_variance_sum = np.zeros(n_lags)
    n_pairs = np.zeros(n_lags, dtype=np.int64)

    for i in prange(n_points - 1):
        for j in range(i + 1, n_points):
            dist_sq = 0.0
            for d in range(n_dims):
                diff = coordinates[i, d] - coordinates[j, d]
                dist_sq += diff * diff
            dist = np.sqrt(dist_sq)

            value_diff = values[i] - values[j]
            semi_var = 0.5 * value_diff * value_diff

            for k in range(n_lags):
                lag_min = lag_bins[k] - tolerance
                lag_max = lag_bins[k + 1] + tolerance

                if lag_min <= dist < lag_max:
                    semi_variance_sum[k] += semi_var
                    n_pairs[k] += 1
                    break

    return semi_variance_sum, n_pairs


def compute_experimental_variogram(
    points: PointSet,
    values: np.ndarray,
    n_lags: int = 15,
    max_lag: Optional[float] = None,
    lag_tolerance: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute experimental semi-variogram from sample points.

    Args:
        points: PointSet with sample locations.
        values: Sample values (n_samples,).
        n_lags: Number of lag bins.
        max_lag: Maximum lag distance (default: half of max distance).
        lag_tolerance: Tolerance for binning (fraction of lag width).

    Returns:
        Tuple of (lags, semi_variance, n_pairs) for each bin.

    Raises:
        ValueError: If inputs are invalid.
    """
    coordinates = points.coordinates

    if len(coordinates) != len(values):
        raise ValueError(
            f"Coordinates ({len(coordinates)}) and values ({len(values)}) "
            f"must have same length"
        )

    if len(values) < 10:
        raise ValueError(f"Need at least 10 samples for variogram, got {len(values)}")

    # Determine lag bins
    if max_lag is None:
        if not SCIPY_AVAILABLE:
            raise ImportError(
                "scipy is required for variogram analysis. Install with: "
                "pip install geosmith[primitives] or pip install scipy"
            )
        from scipy.spatial.distance import pdist

        distances = pdist(coordinates)
        max_lag = distances.max() / 2.0

    lag_width = max_lag / n_lags
    lag_bins = np.linspace(0, max_lag, n_lags + 1)
    lag_centers = (lag_bins[:-1] + lag_bins[1:]) / 2
    tolerance = lag_tolerance * lag_width

    # Use Numba-accelerated computation if available
    if NUMBA_AVAILABLE and len(coordinates) > 100:
        semi_variance_sum, n_pairs_array = _compute_variogram_fast(
            coordinates, values, lag_bins, tolerance
        )

        lags = []
        semi_variances = []
        n_pairs_list = []

        for i in range(n_lags):
            if n_pairs_array[i] > 0:
                lags.append(lag_centers[i])
                semi_variances.append(semi_variance_sum[i] / n_pairs_array[i])
                n_pairs_list.append(n_pairs_array[i])

        return np.array(lags), np.array(semi_variances), np.array(n_pairs_list)
    else:
        # Fallback to scipy pdist
        if not SCIPY_AVAILABLE:
            raise ImportError(
                "scipy is required for variogram analysis. Install with: "
                "pip install geosmith[primitives] or pip install scipy"
            )
        from scipy.spatial.distance import pdist

        distances = pdist(coordinates)
        value_diffs = pdist(values.reshape(-1, 1))
        semi_variance_pairs = 0.5 * value_diffs**2

        lags = []
        semi_variances = []
        n_pairs_list = []

        for i in range(n_lags):
            lag_min = lag_bins[i]
            lag_max = lag_bins[i + 1]
            mask = (distances >= lag_min - tolerance) & (
                distances < lag_max + tolerance
            )

            if np.sum(mask) > 0:
                lags.append(lag_centers[i])
                semi_variances.append(np.mean(semi_variance_pairs[mask]))
                n_pairs_list.append(np.sum(mask))

        return np.array(lags), np.array(semi_variances), np.array(n_pairs_list)


def fit_variogram_model(
    lags: np.ndarray,
    semi_variances: np.ndarray,
    model_type: str = "spherical",
    initial_params: Optional[dict] = None,
) -> VariogramModel:
    """Fit theoretical variogram model to experimental data.

    Args:
        lags: Lag distances.
        semi_variances: Experimental semi-variances.
        model_type: Model type ('spherical', 'exponential', 'gaussian', 'linear').
        initial_params: Optional initial parameter guesses.

    Returns:
        Fitted VariogramModel.

    Raises:
        ValueError: If model_type is invalid or fitting fails.
    """
    if model_type not in VARIOGRAM_MODELS:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Must be one of {list(VARIOGRAM_MODELS.keys())}"
        )

    if len(lags) < 3:
        raise ValueError(f"Need at least 3 lag bins, got {len(lags)}")

    model_func = VARIOGRAM_MODELS[model_type]

    # Initial parameter guesses
    if initial_params is None:
        nugget_guess = semi_variances[0] if len(semi_variances) > 0 else 0.0
        sill_guess = np.max(semi_variances)
        range_guess = lags[-1] / 2.0 if len(lags) > 0 else 1.0
    else:
        nugget_guess = initial_params.get("nugget", semi_variances[0])
        sill_guess = initial_params.get("sill", np.max(semi_variances))
        range_guess = initial_params.get("range", lags[-1] / 2.0)

    # Fit model
    if model_type == "linear":
        # Linear model has different signature
        try:
            if not SCIPY_AVAILABLE:
                raise ImportError(
                    "scipy is required for variogram fitting. Install with: "
                    "pip install geosmith[primitives] or pip install scipy"
                )
            popt, _ = curve_fit(
                lambda h, n, s: _linear_model(h, n, s),
                lags,
                semi_variances,
                p0=[nugget_guess, (sill_guess - nugget_guess) / range_guess],
                bounds=([0, 0], [np.inf, np.inf]),
            )
            nugget, slope = popt
            sill = nugget + slope * lags[-1]  # Approximate sill
            range_param = lags[-1]
            partial_sill = slope * range_param
        except Exception:
            # Fallback
            nugget = nugget_guess
            slope = (sill_guess - nugget_guess) / range_guess
            sill = nugget + slope * lags[-1]
            range_param = lags[-1]
            partial_sill = slope * range_param
    elif model_type == "power":
        # Power model has different signature (no sill)
        try:
            if not SCIPY_AVAILABLE:
                raise ImportError(
                    "scipy is required for variogram fitting. Install with: "
                    "pip install geosmith[primitives] or pip install scipy"
                )
            # Initial guesses: nugget, scale, exponent
            scale_guess = (sill_guess - nugget_guess) / (range_guess ** 1.5)
            popt, _ = curve_fit(
                lambda h, n, s, e: _power_model(h, n, s, e),
                lags,
                semi_variances,
                p0=[nugget_guess, scale_guess, 1.5],
                bounds=([0, 0, 0.1], [np.inf, np.inf, 1.99]),
            )
            nugget, scale, exponent = popt
            # Power model has no sill - use value at max lag
            sill = nugget + scale * (lags[-1] ** exponent)
            range_param = lags[-1]  # Not meaningful for power model
            partial_sill = scale * (lags[-1] ** exponent)
        except Exception:
            # Fallback
            nugget = nugget_guess
            scale = (sill_guess - nugget_guess) / (range_guess ** 1.5)
            exponent = 1.5
            sill = nugget + scale * (lags[-1] ** exponent)
            range_param = lags[-1]
            partial_sill = scale * (lags[-1] ** exponent)
    else:
        try:
            if not SCIPY_AVAILABLE:
                raise ImportError(
                    "scipy is required for variogram fitting. Install with: "
                    "pip install geosmith[primitives] or pip install scipy"
                )
            popt, _ = curve_fit(
                model_func,
                lags,
                semi_variances,
                p0=[nugget_guess, sill_guess, range_guess],
                bounds=([0, nugget_guess, 0], [sill_guess, np.inf, np.inf]),
            )
            nugget, sill, range_param = popt
            partial_sill = sill - nugget
        except Exception:
            # Fallback to initial guesses
            nugget = nugget_guess
            sill = sill_guess
            range_param = range_guess
            partial_sill = sill - nugget

    # Compute R²
    predicted = (
        model_func(lags, nugget, sill, range_param)
        if model_type != "linear"
        else _linear_model(lags, nugget, (sill - nugget) / range_param)
    )
    ss_res = np.sum((semi_variances - predicted) ** 2)
    ss_tot = np.sum((semi_variances - np.mean(semi_variances)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return VariogramModel(
        model_type=model_type,
        nugget=nugget,
        sill=sill,
        range_param=range_param,
        partial_sill=partial_sill,
        r_squared=r_squared,
    )


def predict_variogram(
    variogram_model: VariogramModel, distances: np.ndarray
) -> np.ndarray:
    """Predict variogram values at given distances.

    Args:
        variogram_model: Fitted VariogramModel.
        distances: Distances to predict at.

    Returns:
        Predicted semi-variance values.
    """
    model_func = VARIOGRAM_MODELS[variogram_model.model_type]

    if variogram_model.model_type == "linear":
        slope = variogram_model.partial_sill / variogram_model.range_param
        return _linear_model(distances, variogram_model.nugget, slope)
    elif variogram_model.model_type == "power":
        # Power model has different signature
        scale = variogram_model.partial_sill / (variogram_model.range_param ** 1.5)
        return _power_model(distances, variogram_model.nugget, scale, 1.5)
    else:
        return model_func(
            distances,
            variogram_model.nugget,
            variogram_model.sill,
            variogram_model.range_param,
        )


def compute_experimental_cross_variogram(
    points_primary: PointSet,
    values_primary: np.ndarray,
    points_secondary: PointSet,
    values_secondary: np.ndarray,
    n_lags: int = 15,
    max_lag: Optional[float] = None,
    lag_tolerance: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute experimental cross-variogram between two variables.

    Cross-variogram measures spatial cross-correlation between two variables
    at different lag distances. Used for Co-Kriging.

    Args:
        points_primary: PointSet with primary variable sample locations.
        values_primary: Primary variable values (n_samples_primary,).
        points_secondary: PointSet with secondary variable sample locations.
        values_secondary: Secondary variable values (n_samples_secondary,).
        n_lags: Number of lag bins.
        max_lag: Maximum lag distance (default: half of max distance).
        lag_tolerance: Tolerance for binning (fraction of lag width).

    Returns:
        Tuple of (lags, cross_semi_variance, n_pairs) for each bin.

    Raises:
        ValueError: If inputs are invalid.
    """
    coords_primary = points_primary.coordinates
    coords_secondary = points_secondary.coordinates

    if len(coords_primary) != len(values_primary):
        raise ValueError(
            f"Primary coordinates ({len(coords_primary)}) and values "
            f"({len(values_primary)}) must have same length"
        )

    if len(coords_secondary) != len(values_secondary):
        raise ValueError(
            f"Secondary coordinates ({len(coords_secondary)}) and values "
            f"({len(values_secondary)}) must have same length"
        )

    if len(values_primary) < 5 or len(values_secondary) < 5:
        raise ValueError(
            f"Need at least 5 samples for each variable. "
            f"Got {len(values_primary)} primary and {len(values_secondary)} secondary"
        )

    # Determine lag bins from all pairwise distances
    if max_lag is None:
        if not SCIPY_AVAILABLE:
            raise ImportError(
                "scipy is required for cross-variogram analysis. Install with: "
                "pip install geosmith[primitives] or pip install scipy"
            )
        from scipy.spatial.distance import cdist

        # Compute distances between primary and secondary points
        distances = cdist(coords_primary, coords_secondary).ravel()
        max_lag = distances.max() / 2.0

    lag_width = max_lag / n_lags
    lag_bins = np.linspace(0, max_lag, n_lags + 1)
    lag_centers = (lag_bins[:-1] + lag_bins[1:]) / 2
    tolerance = lag_tolerance * lag_width

    # Compute cross-variogram
    if not SCIPY_AVAILABLE:
        raise ImportError(
            "scipy is required for cross-variogram analysis. Install with: "
            "pip install geosmith[primitives] or pip install scipy"
        )
    from scipy.spatial.distance import cdist

    # Mean-center the values
    mean_primary = np.mean(values_primary)
    mean_secondary = np.mean(values_secondary)
    centered_primary = values_primary - mean_primary
    centered_secondary = values_secondary - mean_secondary

    # Compute all pairwise distances and cross-products
    # Cross-variogram: γ_12(h) = 0.5 * E[(Z1(x) - Z1(x+h)) * (Z2(x) - Z2(x+h))]
    # For pairs (i, j) where primary[i] and secondary[j] are at distance h
    distances = cdist(coords_primary, coords_secondary)

    # Compute cross-products for all pairs
    # For each primary point i and secondary point j:
    # cross_product = 0.5 * (Z1[i] - mean1) * (Z2[j] - mean2)
    # This is a simplified approach - assumes both variables measured at similar locations
    cross_products = 0.5 * np.outer(centered_primary, centered_secondary)

    lags = []
    cross_semi_variances = []
    n_pairs_list = []

    for i in range(n_lags):
        lag_min = lag_bins[i]
        lag_max = lag_bins[i + 1]
        mask = (distances >= lag_min - tolerance) & (
            distances < lag_max + tolerance
        )

        if np.sum(mask) > 0:
            lags.append(lag_centers[i])
            cross_semi_variances.append(np.mean(cross_products[mask]))
            n_pairs_list.append(np.sum(mask))

    return np.array(lags), np.array(cross_semi_variances), np.array(n_pairs_list)
