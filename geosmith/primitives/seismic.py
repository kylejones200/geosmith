"""Seismic processing primitives.

Pure seismic processing operations.
Migrated from geosuite.petro.seismic_processing.
Layer 2: Primitives - Pure operations.
"""

import logging
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# Optional scipy dependency
try:
    from scipy.signal import find_peaks, hilbert

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    hilbert = None  # type: ignore
    find_peaks = None  # type: ignore
    logger.warning(
        "scipy.signal not available. Seismic processing requires scipy. "
        "Install with: pip install geosmith[primitives] or pip install scipy"
    )


def compute_hilbert_attributes(
    trace: Union[np.ndarray, float], dt: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute envelope and unwrapped phase using the Hilbert transform.

    The Hilbert transform is used to compute the analytic signal from a
    real-valued seismic trace, from which the envelope (instantaneous amplitude)
    and phase (instantaneous phase) can be extracted.

    Args:
        trace: Seismic trace (real-valued array).
        dt: Sample interval in seconds.

    Returns:
        Tuple of (time, envelope, phase) arrays:
            - time: Time array (seconds).
            - envelope: Envelope (instantaneous amplitude).
            - phase: Unwrapped phase (radians).

    Example:
        >>> from geosmith.primitives.seismic import compute_hilbert_attributes
        >>>
        >>> trace = np.random.randn(1000)
        >>> dt = 0.004  # 4 ms
        >>> time, envelope, phase = compute_hilbert_attributes(trace, dt)
        >>> print(f"Envelope max: {envelope.max():.2f}")

    Raises:
        ImportError: If scipy.signal is not available.
        ValueError: If trace is empty or invalid.
    """
    if not SCIPY_AVAILABLE:
        raise ImportError(
            "scipy.signal is required for Hilbert transform. "
            "Install with: pip install geosmith[primitives] or pip install scipy"
        )

    trace = np.asarray(trace, dtype=np.float64)

    if len(trace) == 0:
        raise ValueError("Trace cannot be empty")

    # Compute analytic signal using Hilbert transform
    analytic = hilbert(trace)

    # Envelope (instantaneous amplitude)
    envelope = np.abs(analytic)

    # Phase (instantaneous phase), unwrapped to remove discontinuities
    phase = np.unwrap(np.angle(analytic))

    # Time array
    time = np.arange(len(trace)) * dt

    return time, envelope, phase


def estimate_residual_phase(
    phase: Union[np.ndarray, float],
    envelope: Union[np.ndarray, float],
    time: Optional[Union[np.ndarray, float]] = None,
    threshold_pct: float = 75.0,
    distance: int = 10,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Estimate residual phase at envelope peaks.

    This function finds peaks in the envelope and estimates the residual phase
    by computing the circular mean of phase values at those peaks. This is used
    for phase correction of seismic data.

    Args:
        phase: Unwrapped phase array (radians).
        envelope: Envelope (instantaneous amplitude) array.
        time: Time array (seconds). If None, uses sample indices.
        threshold_pct: Percentile threshold for peak detection (0-100), default 75.0.
        distance: Minimum distance between peaks (samples), default 10.

    Returns:
        Tuple of (residual_phase, peak_times, peak_phases):
            - residual_phase: Estimated residual phase (radians), circular mean.
            - peak_times: Time values at envelope peaks.
            - peak_phases: Phase values at envelope peaks.

    Example:
        >>> from geosmith.primitives.seismic import compute_hilbert_attributes, estimate_residual_phase
        >>>
        >>> time, envelope, phase = compute_hilbert_attributes(trace, dt)
        >>> residual, peak_t, peak_p = estimate_residual_phase(phase, envelope, time)
        >>> print(f"Residual phase: {residual:.2f} radians ({np.degrees(residual):.1f}°)")

    Raises:
        ImportError: If scipy.signal is not available.
        ValueError: If inputs are invalid.
    """
    if not SCIPY_AVAILABLE:
        raise ImportError(
            "scipy.signal is required for peak detection. "
            "Install with: pip install geosmith[primitives] or pip install scipy"
        )

    phase = np.asarray(phase, dtype=np.float64)
    envelope = np.asarray(envelope, dtype=np.float64)

    if len(phase) != len(envelope):
        raise ValueError("phase and envelope must have the same length")

    if len(phase) == 0:
        raise ValueError("Input arrays cannot be empty")

    # Find peaks in envelope
    threshold = np.percentile(envelope, threshold_pct)
    peaks, _ = find_peaks(envelope, distance=distance, height=threshold)

    if len(peaks) == 0:
        logger.warning("No peaks found in envelope, returning zero residual phase")
        return 0.0, np.array([]), np.array([])

    # Get phase values at peaks
    peak_phases = phase[peaks]

    # Circular mean of wrapped phase (handles phase wrapping correctly)
    # Convert to complex exponential, average, then get angle
    residual = np.angle(np.mean(np.exp(1j * peak_phases)))

    if time is not None:
        time = np.asarray(time, dtype=np.float64)
        peak_times = time[peaks]
    else:
        peak_times = peaks.astype(float)

    return residual, peak_times, peak_phases


def apply_phase_shift(
    trace: Union[np.ndarray, float],
    phase_shift_rad: float,
    dt: float,
) -> np.ndarray:
    """Apply phase shift to seismic trace.

    Uses Hilbert transform to rotate trace in complex plane.

    Args:
        trace: Seismic trace (real-valued array).
        phase_shift_rad: Phase shift in radians.
        dt: Sample interval in seconds (unused, kept for API consistency).

    Returns:
        Phase-shifted trace.

    Example:
        >>> from geosmith.primitives.seismic import apply_phase_shift
        >>>
        >>> shifted = apply_phase_shift(trace, phase_shift_rad=np.radians(90), dt=0.004)
        >>> print(f"Shifted trace length: {len(shifted)}")
    """
    if not SCIPY_AVAILABLE:
        raise ImportError(
            "scipy.signal is required for phase shift. "
            "Install with: pip install geosmith[primitives] or pip install scipy"
        )

    trace = np.asarray(trace, dtype=np.float64)

    if len(trace) == 0:
        raise ValueError("Trace cannot be empty")

    # Compute analytic signal
    analytic = hilbert(trace)

    # Apply phase rotation
    shifted_analytic = analytic * np.exp(1j * phase_shift_rad)

    # Extract real part (phase-shifted trace)
    shifted = np.real(shifted_analytic)

    return shifted


def correct_trace_phase(
    trace: Union[np.ndarray, float],
    dt: float,
    target_phase_rad: float = 0.0,
    threshold_pct: float = 75.0,
    distance: int = 10,
) -> tuple[np.ndarray, float]:
    """Correct trace phase to target phase.

    Estimates residual phase and corrects trace to target phase (default zero phase).

    Args:
        trace: Seismic trace (real-valued array).
        dt: Sample interval in seconds.
        target_phase_rad: Target phase in radians, default 0.0 (zero phase).
        threshold_pct: Percentile threshold for peak detection, default 75.0.
        distance: Minimum distance between peaks (samples), default 10.

    Returns:
        Tuple of (corrected_trace, applied_phase_shift):
            - corrected_trace: Phase-corrected trace.
            - applied_phase_shift: Phase shift applied (radians).

    Example:
        >>> from geosmith.primitives.seismic import correct_trace_phase
        >>>
        >>> corrected, shift = correct_trace_phase(trace, dt=0.004, target_phase_rad=0.0)
        >>> print(f"Applied phase shift: {np.degrees(shift):.1f}°")
    """
    if not SCIPY_AVAILABLE:
        raise ImportError(
            "scipy.signal is required for phase correction. "
            "Install with: pip install geosmith[primitives] or pip install scipy"
        )

    trace = np.asarray(trace, dtype=np.float64)

    if len(trace) == 0:
        raise ValueError("Trace cannot be empty")

    # Compute Hilbert attributes
    time, envelope, phase = compute_hilbert_attributes(trace, dt)

    # Estimate residual phase
    residual, _, _ = estimate_residual_phase(
        phase, envelope, time, threshold_pct=threshold_pct, distance=distance
    )

    # Calculate phase shift needed to reach target
    phase_shift = target_phase_rad - residual

    # Apply phase shift
    corrected = apply_phase_shift(trace, phase_shift, dt)

    return corrected, phase_shift

