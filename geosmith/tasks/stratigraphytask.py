"""Stratigraphy and change-point detection task.

Migrated from geosuite.stratigraphy.changepoint.
Layer 3: Tasks - User intent translation.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from geosmith.objects.geotable import GeoTable

logger = logging.getLogger(__name__)

# Optional ruptures dependency
try:
    import ruptures as rpt

    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False
    rpt = None  # type: ignore


class StratigraphyTask:
    """Task for automated stratigraphic interpretation using change-point detection.

    Detects formation boundaries in well log data using statistical methods.
    """

    def __init__(self):
        """Initialize StratigraphyTask."""
        pass

    def preprocess_log(
        self,
        log_values: np.ndarray,
        median_window: int = 5,
        detrend_window: int = 100,
    ) -> np.ndarray:
        """Preprocess well log data for change-point detection.

        Applies median filtering to remove spikes (preserves edges) and
        baseline removal to eliminate drift (preserves bed-scale contrasts).

        Args:
            log_values: Raw log values (e.g., GR in API units).
            median_window: Window size for spike removal (default 5 samples).
            detrend_window: Window size for baseline removal in samples (0 to skip).

        Returns:
            Preprocessed log values.

        Example:
            >>> from geosmith.tasks import StratigraphyTask
            >>>
            >>> task = StratigraphyTask()
            >>> gr_clean = task.preprocess_log(gr_raw, median_window=5)

        Raises:
            ValueError: If log_values is empty.
        """
        if len(log_values) == 0:
            raise ValueError("log_values cannot be empty")

        if not isinstance(log_values, np.ndarray):
            log_values = np.array(log_values)

        # Median filter to remove spikes while preserving sharp edges
        log_filtered = median_filter(log_values, size=median_window)

        # Optional detrending (remove long-wavelength drift)
        if detrend_window > 0 and len(log_values) > detrend_window:
            # Compute baseline with large median filter
            baseline = median_filter(log_filtered, size=detrend_window)

            # Remove baseline and restore median to preserve absolute scale
            log_processed = log_filtered - baseline + np.median(log_filtered)
        else:
            log_processed = log_filtered

        return log_processed

    def detect_change_points(
        self,
        log_values: np.ndarray,
        penalty: Optional[float] = None,
        model: str = "l2",
        min_size: int = 3,
        jump: int = 1,
    ) -> np.ndarray:
        """Detect change points using PELT (Pruned Exact Linear Time) algorithm.

        PELT finds the optimal segmentation by minimizing a penalized cost function.
        It guarantees finding the global optimum and runs in near-linear time.

        Args:
            log_values: Preprocessed log values.
            penalty: Penalty value (higher = fewer change points).
                     If None, uses log(n) × variance.
            model: Cost function model ('l2' for mean shift, 'rbf' for kernel-based).
            min_size: Minimum segment length (default 3).
            jump: Subsample factor (1 = no subsampling).

        Returns:
            Array of change point indices (sorted, excluding start/end).

        Raises:
            ImportError: If ruptures library is not installed.
            ValueError: If log_values is empty or invalid.

        Example:
            >>> from geosmith.tasks import StratigraphyTask
            >>>
            >>> task = StratigraphyTask()
            >>> change_points = task.detect_change_points(
            ...     gr_processed, penalty=50.0, model='l2'
            ... )
            >>> formation_tops = depth[change_points]
        """
        if not RUPTURES_AVAILABLE:
            raise ImportError(
                "ruptures library required for change-point detection. "
                "Install with: pip install ruptures"
            )

        if len(log_values) == 0:
            raise ValueError("log_values cannot be empty")

        if not isinstance(log_values, np.ndarray):
            log_values = np.array(log_values)

        # Auto-tune penalty if not provided
        if penalty is None:
            n = len(log_values)
            penalty = np.log(n) * np.var(log_values)
            logger.info(f"Auto-tuned penalty: {penalty:.2f}")

        # Create PELT model
        if model == "l2":
            algo = rpt.Pelt(model="l2", min_size=min_size, jump=jump)
        elif model == "rbf":
            algo = rpt.Pelt(model="rbf", min_size=min_size, jump=jump)
        else:
            algo = rpt.Pelt(model=model, min_size=min_size, jump=jump)

        # Fit and predict
        algo.fit(log_values.reshape(-1, 1))
        change_points = algo.predict(pen=penalty)

        # Remove the final point (always equals length of signal)
        change_points = np.array(change_points[:-1])

        logger.info(f"Detected {len(change_points)} change points")

        return change_points

    def detect_formation_tops(
        self,
        data: GeoTable | pd.DataFrame,
        log_column: str,
        depth_column: str = "DEPTH",
        penalty: Optional[float] = None,
        preprocess: bool = True,
        median_window: int = 5,
        detrend_window: int = 100,
    ) -> pd.DataFrame:
        """Detect formation tops from well log data.

        Complete workflow: preprocessing + change-point detection + depth mapping.

        Args:
            data: GeoTable or DataFrame with log and depth columns.
            log_column: Name of log column to analyze (e.g., 'GR').
            depth_column: Name of depth column (default 'DEPTH').
            penalty: Penalty value for PELT (None = auto-tune).
            preprocess: Whether to preprocess log data (default True).
            median_window: Window size for spike removal.
            detrend_window: Window size for baseline removal.

        Returns:
            DataFrame with formation top depths and indices.

        Example:
            >>> from geosmith.tasks import StratigraphyTask
            >>> from geosmith import GeoTable
            >>>
            >>> task = StratigraphyTask()
            >>> tops = task.detect_formation_tops(
            ...     data=geotable,
            ...     log_column='GR',
            ...     depth_column='DEPTH'
            ... )
        """
        # Extract DataFrame if GeoTable
        if isinstance(data, GeoTable):
            df = data.data.copy()
        else:
            df = data.copy()

        if log_column not in df.columns:
            raise ValueError(
                f"Log column '{log_column}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        if depth_column not in df.columns:
            raise ValueError(
                f"Depth column '{depth_column}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        # Extract log values
        log_values = df[log_column].values

        # Preprocess if requested
        if preprocess:
            log_processed = self.preprocess_log(
                log_values, median_window=median_window, detrend_window=detrend_window
            )
        else:
            log_processed = log_values

        # Detect change points
        change_points = self.detect_change_points(
            log_processed, penalty=penalty, model="l2"
        )

        # Map to depths
        depths = df[depth_column].values
        formation_tops = depths[change_points]

        # Create results DataFrame
        results = pd.DataFrame(
            {
                "index": change_points,
                "depth": formation_tops,
                "log_value": log_processed[change_points],
            }
        )

        logger.info(f"Detected {len(results)} formation tops")

        return results
