"""Decline curve analysis task.

Migrated from geosuite.forecasting.decline_models.
Layer 3: Tasks - User intent translation.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from geosmith.primitives.base import BaseEstimator

logger = logging.getLogger(__name__)

# Optional scipy dependency
try:
    from scipy.optimize import curve_fit
    from scipy.stats import linregress

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    curve_fit = None  # type: ignore
    linregress = None  # type: ignore

# Optional multiprocessing dependency
try:
    import multiprocessing as mp

    MULTIPROCESSING_AVAILABLE = True
except ImportError:
    MULTIPROCESSING_AVAILABLE = False
    mp = None  # type: ignore

# Optional tqdm dependency
try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None  # type: ignore


class DeclineModel(BaseEstimator, ABC):
    """Base class for decline curve models.

    All decline models follow the same interface for fitting and forecasting.
    """

    def __init__(self) -> None:
        """Initialize decline model."""
        super().__init__()
        if not SCIPY_AVAILABLE:
            raise ImportError(
                "scipy is required for decline curve analysis. "
                "Install with: pip install geosmith[primitives] or pip install scipy"
            )
        self.params: Dict[str, float] = {}
        self._fitted = False

    @abstractmethod
    def _rate_function(self, t: np.ndarray, *params: float) -> np.ndarray:
        """Rate function for the decline model.

        Args:
            t: Time array.
            *params: Model parameters.

        Returns:
            Production rate array.
        """
        pass

    @abstractmethod
    def _cumulative_function(self, t: np.ndarray, *params: float) -> np.ndarray:
        """Cumulative production function.

        Args:
            t: Time array.
            *params: Model parameters.

        Returns:
            Cumulative production array.
        """
        pass

    @abstractmethod
    def _estimate_initial_params(
        self, time: np.ndarray, rate: np.ndarray
    ) -> Dict[str, float]:
        """Estimate initial parameters from data.

        Args:
            time: Time array.
            rate: Production rate array.

        Returns:
            Dictionary with initial parameter estimates.
        """
        pass

    @abstractmethod
    def _get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get parameter bounds for optimization.

        Returns:
            Tuple of (lower_bounds, upper_bounds) arrays.
        """
        pass

    def fit(
        self,
        time: Union[np.ndarray, pd.Series],
        rate: Union[np.ndarray, pd.Series],
        initial_params: Optional[Dict[str, float]] = None,
    ) -> "DeclineModel":
        """Fit decline model to production data.

        Args:
            time: Time array (days, months, etc.).
            rate: Production rate (volume/time).
            initial_params: Optional initial parameter guesses.

        Returns:
            self for method chaining.

        Example:
            >>> from geosmith.tasks.declinetask import ExponentialDecline
            >>> import numpy as np
            >>>
            >>> time = np.arange(0, 100, 1)
            >>> rate = 100 * np.exp(-0.01 * time)  # Exponential decline
            >>> model = ExponentialDecline()
            >>> model.fit(time, rate)
            >>> print(f"Fitted params: {model.params}")
        """
        if not SCIPY_AVAILABLE:
            raise ImportError(
                "scipy is required for decline curve fitting. "
                "Install with: pip install scipy"
            )

        time_arr = np.asarray(time, dtype=float)
        rate_arr = np.asarray(rate, dtype=float)

        # Remove invalid values
        valid_mask = (rate_arr > 0) & np.isfinite(rate_arr) & np.isfinite(time_arr)
        time_arr = time_arr[valid_mask]
        rate_arr = rate_arr[valid_mask]

        if len(time_arr) < 3:
            raise ValueError("Need at least 3 valid data points to fit decline model")

        # Normalize time to start at 0
        t0 = time_arr[0]
        time_normalized = time_arr - t0

        # Fit model
        try:
            if initial_params is None:
                initial_params = self._estimate_initial_params(
                    time_normalized, rate_arr
                )

            popt, _ = curve_fit(
                self._rate_function,
                time_normalized,
                rate_arr,
                p0=list(initial_params.values()),
                bounds=self._get_bounds(),
                maxfev=10000,
            )

            # Store parameters
            param_names = list(initial_params.keys())
            self.params = dict(zip(param_names, popt))
            self._fitted = True

            logger.info(f"Fitted {self.__class__.__name__} with params: {self.params}")

        except Exception as e:
            logger.error(f"Error fitting decline model: {e}")
            raise

        return self

    def predict(
        self,
        time: Union[np.ndarray, pd.Series],
        return_cumulative: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict production rate (and optionally cumulative) for given times.

        Args:
            time: Time array for prediction.
            return_cumulative: If True, also return cumulative production,
                default False.

        Returns:
            Production rate array (and cumulative if requested).

        Example:
            >>> from geosmith.tasks.declinetask import ExponentialDecline
            >>> import numpy as np
            >>>
            >>> model = ExponentialDecline()
            >>> model.fit(time_train, rate_train)
            >>> predictions = model.predict(time_future)
            >>> rate, cumulative = model.predict(time_future, return_cumulative=True)
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")

        time_arr = np.asarray(time, dtype=float)
        time_normalized = time_arr - time_arr[0]

        rate = self._rate_function(time_normalized, *list(self.params.values()))

        if return_cumulative:
            cumulative = self._cumulative_function(
                time_normalized, *list(self.params.values())
            )
            return rate, cumulative

        return rate

    def forecast(
        self,
        n_periods: int,
        period_length: float = 1.0,
        start_time: Optional[float] = None,
    ) -> pd.DataFrame:
        """Forecast production for future periods.

        Args:
            n_periods: Number of periods to forecast.
            period_length: Length of each period (same units as training data),
                default 1.0.
            start_time: Optional start time for forecast (uses 0 if not specified).

        Returns:
            DataFrame with columns: 'time', 'rate', 'cumulative'.

        Example:
            >>> from geosmith.tasks.declinetask import HyperbolicDecline
            >>> import numpy as np
            >>>
            >>> model = HyperbolicDecline()
            >>> model.fit(time_train, rate_train)
            >>> forecast = model.forecast(n_periods=12, period_length=1.0)
            >>> print(f"Forecast rates: {forecast['rate'].values}")
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before forecasting")

        if start_time is None:
            # Use 0 (normalized time)
            start_time = 0.0

        forecast_times = np.arange(n_periods) * period_length + start_time
        rate, cumulative = self.predict(forecast_times, return_cumulative=True)

        return pd.DataFrame(
            {"time": forecast_times, "rate": rate, "cumulative": cumulative}
        )


class ExponentialDecline(DeclineModel):
    """Exponential decline model: q(t) = q_i * exp(-D_i * t).

    Where:
    - q_i: Initial production rate
    - D_i: Decline rate (1/time)
    """

    def _rate_function(self, t: np.ndarray, qi: float, di: float) -> np.ndarray:
        """Exponential decline rate function."""
        return qi * np.exp(-di * t)

    def _cumulative_function(self, t: np.ndarray, qi: float, di: float) -> np.ndarray:
        """Exponential decline cumulative function."""
        return (qi / di) * (1 - np.exp(-di * t))

    def _estimate_initial_params(
        self, time: np.ndarray, rate: np.ndarray
    ) -> Dict[str, float]:
        """Estimate initial parameters for exponential decline."""
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for parameter estimation")

        # Use linear regression on log(rate) vs time
        valid_mask = rate > 0
        if np.sum(valid_mask) < 2:
            return {"qi": float(rate[0]), "di": 0.01}

        log_rate = np.log(rate[valid_mask])
        time_valid = time[valid_mask]

        slope, intercept, _, _, _ = linregress(time_valid, log_rate)

        qi = np.exp(intercept)
        di = -slope if slope < 0 else 0.01

        return {"qi": float(qi), "di": float(di)}

    def _get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Parameter bounds for exponential decline."""
        return (
            np.array([0.0, 0.0]),  # qi > 0, di > 0
            np.array([np.inf, np.inf]),
        )


class HyperbolicDecline(DeclineModel):
    """Hyperbolic decline model: q(t) = q_i / (1 + b * D_i * t)^(1/b).

    Where:
    - q_i: Initial production rate
    - D_i: Initial decline rate (1/time)
    - b: Hyperbolic exponent (0 < b < 1)
    """

    def _rate_function(
        self, t: np.ndarray, qi: float, di: float, b: float
    ) -> np.ndarray:
        """Hyperbolic decline rate function."""
        return qi / np.power(1 + b * di * t, 1.0 / b)

    def _cumulative_function(
        self, t: np.ndarray, qi: float, di: float, b: float
    ) -> np.ndarray:
        """Hyperbolic decline cumulative function."""
        if b == 1.0:
            # Harmonic case
            return (qi / di) * np.log(1 + di * t)
        else:
            return (qi / ((1 - b) * di)) * (1 - np.power(1 + b * di * t, (b - 1) / b))

    def _estimate_initial_params(
        self, time: np.ndarray, rate: np.ndarray
    ) -> Dict[str, float]:
        """Estimate initial parameters for hyperbolic decline."""
        qi = float(rate[0])
        di = 0.01
        b = 0.5  # Typical value

        return {"qi": qi, "di": di, "b": b}

    def _get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Parameter bounds for hyperbolic decline."""
        return (
            np.array([0.0, 0.0, 0.01]),  # qi > 0, di > 0, 0.01 < b < 1
            np.array([np.inf, np.inf, 0.99]),
        )


class HarmonicDecline(DeclineModel):
    """Harmonic decline model: q(t) = q_i / (1 + D_i * t).

    Where:
    - q_i: Initial production rate
    - D_i: Decline rate (1/time)

    This is a special case of hyperbolic decline with b = 1.
    """

    def _rate_function(self, t: np.ndarray, qi: float, di: float) -> np.ndarray:
        """Harmonic decline rate function."""
        return qi / (1 + di * t)

    def _cumulative_function(self, t: np.ndarray, qi: float, di: float) -> np.ndarray:
        """Harmonic decline cumulative function."""
        return (qi / di) * np.log(1 + di * t)

    def _estimate_initial_params(
        self, time: np.ndarray, rate: np.ndarray
    ) -> Dict[str, float]:
        """Estimate initial parameters for harmonic decline."""
        qi = float(rate[0])
        # Estimate di from rate ratio
        if len(rate) > 1 and rate[-1] > 0:
            di = (qi / rate[-1] - 1) / time[-1] if time[-1] > 0 else 0.01
        else:
            di = 0.01

        return {"qi": qi, "di": float(di)}

    def _get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Parameter bounds for harmonic decline."""
        return (
            np.array([0.0, 0.0]),  # qi > 0, di > 0
            np.array([np.inf, np.inf]),
        )


def fit_decline_model(
    time: Union[np.ndarray, pd.Series],
    rate: Union[np.ndarray, pd.Series],
    model_type: str = "hyperbolic",
) -> DeclineModel:
    """Fit a decline curve model to production data.

    Convenience function for fitting decline models.

    Args:
        time: Time array.
        rate: Production rate.
        model_type: Type of decline model ('exponential', 'hyperbolic',
            'harmonic'), default 'hyperbolic'.

    Returns:
        Fitted decline model.

    Example:
        >>> from geosmith.tasks.declinetask import fit_decline_model
        >>> import numpy as np
        >>>
        >>> time = np.arange(0, 100, 1)
        >>> rate = 100 / (1 + 0.01 * time)  # Harmonic decline
        >>> model = fit_decline_model(time, rate, model_type='harmonic')
        >>> forecast = model.forecast(n_periods=12)
        >>> print(f"Forecast rates: {forecast['rate'].head()}")
    """
    model_map = {
        "exponential": ExponentialDecline,
        "hyperbolic": HyperbolicDecline,
        "harmonic": HarmonicDecline,
    }

    if model_type not in model_map:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Choose from {list(model_map.keys())}"
        )

    model = model_map[model_type]()
    model.fit(time, rate)

    return model


def forecast_production(
    model: DeclineModel,
    n_periods: int,
    period_length: float = 1.0,
) -> pd.DataFrame:
    """Forecast production using a fitted decline model.

    Convenience function for forecasting.

    Args:
        model: Fitted decline model.
        n_periods: Number of periods to forecast.
        period_length: Length of each period, default 1.0.

    Returns:
        DataFrame with forecast results.

    Example:
        >>> from geosmith.tasks.declinetask import (
        ...     fit_decline_model, forecast_production
        ... )
        >>> import numpy as np
        >>>
        >>> time = np.arange(0, 50, 1)
        >>> rate = 100 * np.exp(-0.01 * time)
        >>> model = fit_decline_model(time, rate, model_type='exponential')
        >>> forecast = forecast_production(model, n_periods=12, period_length=1.0)
        >>> print(f"Forecast total: {forecast['cumulative'].iloc[-1]:.2f}")
    """
    return model.forecast(n_periods, period_length)


def process_wells_parallel(
    well_data_list: List[Tuple[Any, pd.DataFrame]],
    model_type: str = "hyperbolic",
    date_col: str = "date",
    production_col: str = "production",
    n_jobs: Optional[int] = None,
    batch_size: int = 1000,
    min_data_points: int = 12,
) -> List[Dict[str, Any]]:
    """Process multiple wells in parallel using decline curve analysis.

    This function uses multiprocessing to analyze large datasets efficiently.
    Useful for processing thousands of wells.

    Args:
        well_data_list: List of (well_id, well_dataframe) tuples.
        model_type: Decline model type ('exponential', 'hyperbolic',
            'harmonic'), default 'hyperbolic'.
        date_col: Name of date column in well dataframes, default 'date'.
        production_col: Name of production column in well dataframes,
            default 'production'.
        n_jobs: Number of parallel workers (default: min(cpu_count(), 8)), optional.
        batch_size: Batch size for processing (helps manage memory), default 1000.
        min_data_points: Minimum data points required for analysis, default 12.

    Returns:
        List of analysis results for each well:
            - 'well_id': Well identifier
            - 'model_type': Decline model type used
            - 'n_data_points': Number of data points used
            - 'date_start', 'date_end': Date range
            - 'historical_mean', 'historical_total': Historical statistics
            - 'forecast_mean', 'forecast_total': Forecast statistics
            - 'parameters': Model parameters
            - 'status': 'success' or error message

    Example:
        >>> from geosmith.tasks.declinetask import process_wells_parallel
        >>> import pandas as pd
        >>>
        >>> well_data = [
        ...     ('well_1', df1),
        ...     ('well_2', df2),
        ...     ('well_3', df3)
        ... ]
        >>> results = process_wells_parallel(
        ...     well_data,
        ...     model_type='hyperbolic',
        ...     date_col='date',
        ...     production_col='oil'
        ... )
        >>> print(f"Processed {len(results)} wells")
    """
    if not MULTIPROCESSING_AVAILABLE:
        raise ImportError(
            "multiprocessing is required for parallel processing. "
            "This is typically available in Python standard library."
        )

    if not SCIPY_AVAILABLE:
        raise ImportError(
            "scipy is required for decline curve analysis. "
            "Install with: pip install scipy"
        )

    def analyze_well(args: Tuple[Any, pd.DataFrame]) -> Optional[Dict[str, Any]]:
        """Analyze a single well (used by multiprocessing)."""
        well_id, well_df = args

        try:
            # Sort by date
            if date_col in well_df.columns:
                well_df = well_df.sort_values(date_col)

            # Extract time series
            if date_col in well_df.columns and production_col in well_df.columns:
                dates = pd.to_datetime(well_df[date_col], errors="coerce")
                production = well_df[production_col].fillna(0)

                # Remove invalid values
                valid_mask = dates.notna() & (production > 0)
                if valid_mask.sum() < min_data_points:
                    return None

                dates_valid = dates[valid_mask]
                production_valid = production[valid_mask]

                # Resample to monthly if needed
                series = pd.Series(production_valid.values, index=dates_valid)
                series = series.resample("MS").sum()
                series = series[series > 0]

                if len(series) < min_data_points:
                    return None

                # Fit decline model
                model = fit_decline_model(
                    np.arange(len(series)),
                    series.values,
                    model_type=model_type,
                )

                # Extract parameters
                params = model.params.copy()

                # Generate forecast
                forecast_df = model.forecast(n_periods=12, period_length=1.0)

                return {
                    "well_id": well_id,
                    "model_type": model_type,
                    "n_data_points": len(series),
                    "date_start": series.index.min(),
                    "date_end": series.index.max(),
                    "historical_mean": float(series.mean()),
                    "historical_total": float(series.sum()),
                    "forecast_mean": float(forecast_df["rate"].mean()),
                    "forecast_total": float(forecast_df["rate"].sum()),
                    "parameters": params,
                    "status": "success",
                }

            else:
                return None

        except Exception as e:
            logger.warning(f"Error analyzing well {well_id}: {e}")
            return {
                "well_id": well_id,
                "status": f"error: {str(e)[:100]}",
            }

    # Set up parallel processing
    n_jobs = n_jobs or min(mp.cpu_count(), 8)

    logger.info(f"Processing {len(well_data_list)} wells with {n_jobs} workers")
    logger.info(f"Batch size: {batch_size}, Min data points: {min_data_points}")

    # Process in batches
    results = []
    batches = range(0, len(well_data_list), batch_size)

    if TQDM_AVAILABLE:
        batches = tqdm(batches, desc="Processing batches")

    for i in batches:
        batch = well_data_list[i : i + batch_size]

        with mp.Pool(n_jobs) as pool:
            batch_results = pool.map(analyze_well, batch)

        # Filter out None results
        batch_results = [r for r in batch_results if r is not None]
        results.extend(batch_results)

    logger.info(f"Successfully analyzed {len(results)} wells")

    return results
