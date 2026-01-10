"""Geomechanics: Stress inversion from observations

Pure geomechanics operations - inversion module.
Migrated from geosuite.geomech.
Layer 2: Primitives - Pure operations.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

from geosmith.primitives.geomechanics._common import (
    NUMBA_AVAILABLE,
    PANDAS_AVAILABLE,
    njit,
    pd,
)

if TYPE_CHECKING:
    import pandas as pd

# Optional scipy for optimization
try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def invert_stress_from_breakout(
    breakout_width: Union[np.ndarray, float],
    breakout_azimuth: Union[np.ndarray, float],
    depth: Union[np.ndarray, float],
    sv: Union[np.ndarray, float],
    pp: Union[np.ndarray, float],
    ucs: float = 50.0,
    poisson: float = 0.25,
    wellbore_azimuth: float = 0.0,
    wellbore_inclination: float = 0.0,
    method: str = "analytical",
) -> dict[str, Union[np.ndarray, float]]:
    """Invert stress magnitudes from breakout observations.

    Uses breakout width and azimuth to constrain SHmax and Shmin.

    Args:
        breakout_width: Breakout width in degrees.
        breakout_azimuth: Breakout azimuth in degrees (from north).
        depth: Depth (m).
        sv: Vertical stress (MPa).
        pp: Pore pressure (MPa).
        ucs: Unconfined compressive strength (MPa), default 50.0.
        poisson: Poisson's ratio, default 0.25.
        wellbore_azimuth: Wellbore azimuth in degrees, default 0.0.
        wellbore_inclination: Wellbore inclination in degrees, default 0.0.
        method: Inversion method ('optimization' or 'analytical'), default 'analytical'.

    Returns:
        Dictionary with estimated SHmax, Shmin, and stress_ratio.

    Example:
        >>> from geosmith.primitives.geomechanics import invert_stress_from_breakout
        >>>
        >>> result = invert_stress_from_breakout(
        ...     breakout_width=30.0, breakout_azimuth=45.0,
        ...     depth=2000.0, sv=50.0, pp=20.0
        ... )
        >>> print(f"Estimated SHmax: {result['shmax']:.1f} MPa")
    """
    breakout_width = np.asarray(breakout_width, dtype=float)
    breakout_azimuth = np.asarray(breakout_azimuth, dtype=float)
    depth = np.asarray(depth, dtype=float)
    sv = np.asarray(sv, dtype=float)
    pp = np.asarray(pp, dtype=float)
    
    # Ensure arrays are 1D (handle scalar inputs)
    breakout_width = np.atleast_1d(breakout_width)
    breakout_azimuth = np.atleast_1d(breakout_azimuth)
    depth = np.atleast_1d(depth)
    sv = np.atleast_1d(sv)
    pp = np.atleast_1d(pp)

    if method == "optimization":
        return _invert_stress_optimization(
            breakout_width,
            breakout_azimuth,
            depth,
            sv,
            pp,
            ucs,
            poisson,
            wellbore_azimuth,
            wellbore_inclination,
        )
    else:
        return _invert_stress_analytical(
            breakout_width, depth, sv, pp, ucs, poisson
        )

def _invert_stress_optimization(
    breakout_width: np.ndarray,
    breakout_azimuth: np.ndarray,
    depth: np.ndarray,
    sv: np.ndarray,
    pp: np.ndarray,
    ucs: float,
    poisson: float,
    wellbore_azimuth: float,
    wellbore_inclination: float,
) -> dict[str, np.ndarray]:
    """Invert stress using optimization approach (requires scipy)."""
    try:
        from scipy.optimize import minimize

        SCIPY_AVAILABLE = True
    except ImportError:
        SCIPY_AVAILABLE = False

    if not SCIPY_AVAILABLE:
        raise ImportError(
            "scipy is required for optimization-based stress inversion. "
            "Install with: pip install geosmith[primitives] or pip install scipy. "
            "Alternatively, use method='analytical'"
        )

    results = []

    for i in range(len(breakout_width)):

        def objective(params):
            shmax, shmin = params
            shmax_eff = shmax - pp[i]
            shmin_eff = shmin - pp[i]
            sv_eff = sv[i] - pp[i]

            # Simplified breakout width model
            # Breakout occurs when tangential stress exceeds UCS
            stress_diff = shmax_eff - shmin_eff
            with np.errstate(invalid="ignore"):
                predicted_width = np.degrees(
                    2 * np.arcsin(np.clip(ucs / (2 * stress_diff), -1, 1))
                )

            # Minimize difference between observed and predicted
            width_error = (predicted_width - breakout_width[i]) ** 2

            # Add constraint penalties
            penalty = 0.0
            if shmax < shmin:
                penalty += 1000
            if shmax > sv[i] * 1.5:
                penalty += 1000
            if shmin < pp[i]:
                penalty += 1000

            return width_error + penalty

        # Initial guess: typical stress ratios
        initial_guess = [sv[i] * 1.1, sv[i] * 0.7]
        bounds = [(pp[i], sv[i] * 1.5), (pp[i], sv[i])]

        try:
            result = minimize(
                objective, initial_guess, bounds=bounds, method="L-BFGS-B"
            )
            if result.success:
                shmax_est, shmin_est = result.x
            else:
                shmax_est, shmin_est = initial_guess
        except Exception:
            shmax_est, shmin_est = initial_guess

        stress_ratio = (
            (shmax_est - pp[i]) / (sv[i] - pp[i])
            if (sv[i] - pp[i]) > 0
            else 1.0
        )

        results.append(
            {"shmax": shmax_est, "shmin": shmin_est, "stress_ratio": stress_ratio}
        )

    return {
        "shmax": np.array([r["shmax"] for r in results]),
        "shmin": np.array([r["shmin"] for r in results]),
        "stress_ratio": np.array([r["stress_ratio"] for r in results]),
    }

def _invert_stress_analytical(
    breakout_width: np.ndarray,
    depth: np.ndarray,
    sv: np.ndarray,
    pp: np.ndarray,
    ucs: float,
    poisson: float,
) -> dict[str, np.ndarray]:
    """Invert stress using analytical approach (simplified)."""
    sv_eff = sv - pp

    # Simplified model: breakout width relates to stress difference
    # Assuming breakout occurs when tangential stress = UCS
    with np.errstate(divide="ignore", invalid="ignore"):
        stress_diff = ucs / (2 * np.sin(np.radians(breakout_width / 2)))
    stress_diff = np.clip(stress_diff, 0, sv_eff)
    stress_diff = np.where(np.isnan(stress_diff), sv_eff * 0.2, stress_diff)

    # Estimate Shmin from Poisson's ratio
    shmin = pp + poisson / (1 - poisson) * sv_eff

    # Estimate SHmax from stress difference
    shmax = shmin + stress_diff

    # Ensure physical constraints
    shmax = np.clip(shmax, pp, sv * 1.5)
    shmin = np.clip(shmin, pp, sv)

    stress_ratio = (shmax - pp) / sv_eff
    stress_ratio = np.clip(stress_ratio, 0.5, 2.0)

    return {"shmax": shmax, "shmin": shmin, "stress_ratio": stress_ratio}

def invert_stress_from_dif(
    dif_azimuth: Union[np.ndarray, float],
    depth: Union[np.ndarray, float],
    sv: Union[np.ndarray, float],
    pp: Union[np.ndarray, float],
    tensile_strength: float = 5.0,
    poisson: float = 0.25,
) -> dict[str, Union[np.ndarray, float]]:
    """Invert stress magnitudes from drilling-induced fracture (DIF) observations.

    DIFs form perpendicular to Shmin direction.

    Args:
        dif_azimuth: DIF azimuth in degrees (from north).
        depth: Depth (m).
        sv: Vertical stress (MPa).
        pp: Pore pressure (MPa).
        tensile_strength: Tensile strength (MPa), default 5.0.
        poisson: Poisson's ratio, default 0.25.

    Returns:
        Dictionary with estimated SHmax, Shmin, stress_ratio, and shmin_azimuth.

    Example:
        >>> from geosmith.primitives.geomechanics import invert_stress_from_dif
        >>>
        >>> result = invert_stress_from_dif(
        ...     dif_azimuth=135.0, depth=2000.0, sv=50.0, pp=20.0
        ... )
        >>> print(f"Estimated Shmin azimuth: {result['shmin_azimuth']:.1f}°")
    """
    dif_azimuth = np.asarray(dif_azimuth, dtype=float)
    depth = np.asarray(depth, dtype=float)
    sv = np.asarray(sv, dtype=float)
    pp = np.asarray(pp, dtype=float)

    sv_eff = sv - pp

    # DIF forms when wellbore pressure exceeds minimum stress + tensile strength
    # Shmin is approximately perpendicular to DIF azimuth
    # Simplified: estimate Shmin from vertical stress
    shmin = pp + poisson / (1 - poisson) * sv_eff

    # SHmax is constrained by DIF formation
    # DIF forms when: Pw > Shmin + T0
    # This gives lower bound on Shmin
    shmin_lower = pp + tensile_strength * 0.5

    shmin = np.maximum(shmin, shmin_lower)
    shmin = np.clip(shmin, pp, sv)

    # Estimate SHmax (typically 1.1-1.3x Shmin for normal faulting)
    shmax = shmin * 1.2
    shmax = np.clip(shmax, sv, sv * 1.5)

    stress_ratio = (shmax - pp) / sv_eff
    stress_ratio = np.clip(stress_ratio, 0.5, 2.0)

    return {
        "shmax": shmax,
        "shmin": shmin,
        "stress_ratio": stress_ratio,
        "shmin_azimuth": (dif_azimuth + 90) % 360,
    }

def invert_stress_combined(
    depth: Union[np.ndarray, float],
    sv: Union[np.ndarray, float],
    pp: Union[np.ndarray, float],
    breakout_data: Optional[dict[str, np.ndarray]] = None,
    dif_data: Optional[dict[str, np.ndarray]] = None,
    ucs: float = 50.0,
    tensile_strength: float = 5.0,
    poisson: float = 0.25,
) -> dict[str, Union[np.ndarray, float, str]]:
    """Invert stress magnitudes from combined breakout and DIF observations.

    Combines constraints from both failure types for more robust estimates.

    Args:
        depth: Depth array (m).
        sv: Vertical stress array (MPa).
        pp: Pore pressure array (MPa).
        breakout_data: Dictionary with 'width' and 'azimuth' keys, optional.
        dif_data: Dictionary with 'azimuth' key, optional.
        ucs: Unconfined compressive strength (MPa), default 50.0.
        tensile_strength: Tensile strength (MPa), default 5.0.
        poisson: Poisson's ratio, default 0.25.

    Returns:
        Dictionary with estimated SHmax, Shmin, stress_ratio, and confidence.

    Example:
        >>> from geosmith.primitives.geomechanics import invert_stress_combined
        >>>
        >>> result = invert_stress_combined(
        ...     depth=2000.0, sv=50.0, pp=20.0,
        ...     breakout_data={'width': 30.0, 'azimuth': 45.0},
        ...     dif_data={'azimuth': 135.0}
        ... )
        >>> print(f"Confidence: {result['confidence']}")
    """

    depth = np.asarray(depth, dtype=float)
    sv = np.asarray(sv, dtype=float)
    pp = np.asarray(pp, dtype=float)

    results_breakout = None
    results_dif = None

    if breakout_data is not None:
        results_breakout = _invert_stress_analytical(
            np.asarray(breakout_data["width"], dtype=float),
            depth,
            sv,
            pp,
            ucs,
            poisson,
        )

    if dif_data is not None:
        results_dif = invert_stress_from_dif(
            np.asarray(dif_data["azimuth"], dtype=float),
            depth,
            sv,
            pp,
            tensile_strength,
            poisson,
        )

    # Combine results
    if results_breakout is not None and results_dif is not None:
        # Weighted average (breakout typically more reliable)
        weight_breakout = 0.7
        weight_dif = 0.3

        shmax = (
            weight_breakout * results_breakout["shmax"]
            + weight_dif * results_dif["shmax"]
        )
        shmin = (
            weight_breakout * results_breakout["shmin"]
            + weight_dif * results_dif["shmin"]
        )

        confidence = "high"
    elif results_breakout is not None:
        shmax = results_breakout["shmax"]
        shmin = results_breakout["shmin"]
        confidence = "medium"
    elif results_dif is not None:
        shmax = results_dif["shmax"]
        shmin = results_dif["shmin"]
        confidence = "low"
    else:
        raise ValueError(
            "At least one of breakout_data or dif_data must be provided"
        )

    sv_eff = sv - pp
    stress_ratio = (shmax - pp) / sv_eff
    stress_ratio = np.clip(stress_ratio, 0.5, 2.0)

    return {
        "shmax": shmax,
        "shmin": shmin,
        "stress_ratio": stress_ratio,
        "confidence": confidence,
    }

