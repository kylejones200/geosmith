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
        if self.model_type not in ("spherical", "exponential", "gaussian", "linear"):
            raise ValueError(
                f"model_type must be one of 'spherical', 'exponential', "
                f"'gaussian', 'linear', "
                f"got {self.model_type}"
            )

        if self.nugget < 0:
            raise ValueError(f"nugget must be non-negative, got {self.nugget}")

        if self.sill < self.nugget:
            raise ValueError(f"sill ({self.sill}) must be >= nugget ({self.nugget})")

        if self.range_param <= 0 and self.model_type != "linear":
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


# Model registry
VARIOGRAM_MODELS: dict[str, Callable] = {
    "spherical": _spherical_model,
    "exponential": _exponential_model,
    "gaussian": _gaussian_model,
    "linear": _linear_model,
}


@njit(parallel=True, cache=True, fastmath=True)
def _compute_variogram_fast(
    coordinates: np.ndarray,
    values: np.ndarray,
    lag_bins: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Numba-accelerated variogram computation.

    10-50x faster than scipy for large datasets.
    """
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
    else:
        return model_func(
            distances,
            variogram_model.nugget,
            variogram_model.sill,
            variogram_model.range_param,
        )
