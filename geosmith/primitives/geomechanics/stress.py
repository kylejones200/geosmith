"""Geomechanics: Basic stress calculations

Pure geomechanics operations - stress module.
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


def calculate_effective_stress(
    sv: Union[np.ndarray, float],
    pp: Union[np.ndarray, float],
    biot: float = 1.0,
) -> np.ndarray:
    """Calculate vertical effective stress.

    σ'v = Sv - α * Pp

    Args:
        sv: Overburden stress (MPa).
        pp: Pore pressure (MPa).
        biot: Biot coefficient (typically 0.7-1.0).

    Returns:
        Effective stress (MPa).

    Example:
        >>> from geosmith.primitives.geomechanics import calculate_effective_stress
        >>>
        >>> sv_eff = calculate_effective_stress(sv=50.0, pp=20.0, biot=1.0)
        >>> print(f"Effective stress: {sv_eff:.1f} MPa")
    """
    return np.asarray(sv, dtype=float) - biot * np.asarray(pp, dtype=float)

def calculate_stress_ratio(
    shmin: Union[np.ndarray, float],
    sv: Union[np.ndarray, float],
) -> np.ndarray:
    """Calculate minimum horizontal stress ratio.

    K = Shmin / Sv

    Args:
        shmin: Minimum horizontal stress (MPa).
        sv: Vertical stress (MPa).

    Returns:
        Stress ratio (dimensionless).

    Example:
        >>> from geosmith.primitives.geomechanics import calculate_stress_ratio
        >>>
        >>> k = calculate_stress_ratio(shmin=40.0, sv=50.0)
        >>> print(f"Stress ratio: {k:.2f}")
    """
    shmin = np.asarray(shmin, dtype=float)
    sv = np.asarray(sv, dtype=float)

    # Avoid division by zero
    sv = np.where(sv == 0, np.nan, sv)

    return shmin / sv

def stress_polygon_limits(
    sv: Union[np.ndarray, float],
    pp: Union[np.ndarray, float],
    shmin: Optional[Union[np.ndarray, float]] = None,
    mu: float = 0.6,
    cohesion: float = 0.0,
) -> dict[str, tuple[Union[np.ndarray, float], Optional[Union[np.ndarray, float]]]]:
    """Calculate stress polygon limits for faulting regime determination.

    Returns allowable ranges for SHmax based on faulting theory using Mohr-Coulomb
    failure criterion.

    Args:
        sv: Vertical stress (MPa).
        pp: Pore pressure (MPa).
        shmin: Minimum horizontal stress (MPa), optional.
        mu: Coefficient of friction (typically 0.6-1.0).
        cohesion: Cohesion (MPa), typically 0.

    Returns:
        Dictionary with stress limits for each faulting regime:
        - 'normal': (min, max) for normal faulting
        - 'strike_slip': (min, max) for strike-slip faulting
        - 'reverse': (min, max) for reverse/thrust faulting (max may be None if shmin not provided).

    Example:
        >>> from geosmith.primitives.geomechanics import stress_polygon_limits
        >>>
        >>> limits = stress_polygon_limits(sv=50.0, pp=20.0, shmin=30.0)
        >>> print(f"Normal faulting range: {limits['normal']}")
        >>> print(f"Strike-slip range: {limits['strike_slip']}")
        >>> print(f"Reverse faulting range: {limits['reverse']}")
    """
    sv = np.asarray(sv, dtype=float)
    pp = np.asarray(pp, dtype=float)

    # Broadcast scalars
    if sv.ndim == 0:
        sv = sv.reshape(1)
    if pp.ndim == 0:
        pp = pp.reshape(1)

    if len(sv) != len(pp):
        raise ValueError(
            f"sv ({len(sv)}) and pp ({len(pp)}) must have same length"
        )

    # Mohr-Coulomb failure criterion
    q = np.sqrt(mu**2 + 1) + mu

    # Effective stresses
    sv_eff = sv - pp

    # Normal faulting: Sv > SHmax > Shmin
    # SHmax_max = Sv
    # SHmax_min = (Sv - C) / q + Pp
    nf_max = sv
    nf_min = (sv_eff - cohesion) / q + pp

    # Strike-slip faulting: SHmax > Sv > Shmin
    # SHmax_max = q * (Sv - Pp) + C + Pp
    # SHmax_min = Sv
    ss_min = sv
    ss_max = q * sv_eff + cohesion + pp

    # Reverse/Thrust faulting: SHmax > Shmin > Sv
    # SHmax_min = q * (Sv - Pp) + C + Pp (transition from strike-slip to reverse)
    # Note: rf_min equals ss_max at the transition point (they're equal by definition)
    # If Shmin is known: SHmax_max = q * (Shmin - Pp) + C + Pp
    rf_min = q * sv_eff + cohesion + pp  # Equal to ss_max at transition
    if shmin is not None:
        shmin = np.asarray(shmin, dtype=float)
        if shmin.ndim == 0:
            shmin = shmin.reshape(1)
        if len(shmin) != len(sv):
            raise ValueError(
                f"shmin ({len(shmin)}) must have same length as sv ({len(sv)})"
            )
        shmin_eff = shmin - pp
        rf_max = q * shmin_eff + cohesion + pp
    else:
        rf_max = None

    # Convert back to scalars if input was scalar
    if sv.size == 1:
        nf_min = float(nf_min[0])
        nf_max = float(nf_max[0])
        ss_min = float(ss_min[0])
        ss_max = float(ss_max[0])
        if rf_max is not None:
            rf_max = float(rf_max[0])
        rf_min = float(rf_min[0])

    return {
        "normal": (nf_min, nf_max),
        "strike_slip": (ss_min, ss_max),
        "reverse": (rf_min, rf_max),
    }

def estimate_shmin_from_poisson(
    sv: Union[np.ndarray, float],
    pp: Union[np.ndarray, float],
    nu: float = 0.25,
    biot: float = 1.0,
) -> np.ndarray:
    """Estimate minimum horizontal stress from Poisson's ratio.

    Shmin = (ν / (1 - ν)) * (Sv - α*Pp) + α*Pp

    Args:
        sv: Vertical stress (MPa).
        pp: Pore pressure (MPa).
        nu: Poisson's ratio (default 0.25).
        biot: Biot coefficient (default 1.0).

    Returns:
        Minimum horizontal stress (MPa) as numpy array.

    Example:
        >>> from geosmith.primitives.geomechanics import estimate_shmin_from_poisson
        >>>
        >>> shmin = estimate_shmin_from_poisson(sv=50.0, pp=20.0, nu=0.25)
        >>> print(f"Minimum horizontal stress: {shmin:.1f} MPa")
    """
    sv = np.asarray(sv, dtype=float)
    pp = np.asarray(pp, dtype=float)
    
    # Ensure arrays are 1D (handle scalar inputs)
    sv = np.atleast_1d(sv)
    pp = np.atleast_1d(pp)

    if len(sv) == 0 or len(pp) == 0:
        raise ValueError("Stress and pressure arrays must not be empty")

    if len(sv) != len(pp):
        raise ValueError("Stress and pressure arrays must have same length")

    # Avoid division by zero for Poisson's ratio
    if nu >= 1.0 or nu <= 0.0:
        raise ValueError(f"Poisson's ratio must be in (0, 1). Got: {nu}")

    sigma_v_eff = sv - biot * pp
    shmin = (nu / (1 - nu)) * sigma_v_eff + biot * pp

    return shmin

def friction_coefficient_ratio(mu: float) -> float:
    """Calculate the friction coefficient ratio for faulting calculations.

    fμ = ((μ²+1)^0.5 + μ)²

    Args:
        mu: Friction coefficient.

    Returns:
        Friction coefficient ratio.

    Example:
        >>> from geosmith.primitives.geomechanics import friction_coefficient_ratio
        >>>
        >>> f_mu = friction_coefficient_ratio(mu=0.6)
        >>> print(f"Friction ratio: {f_mu:.3f}")
    """
    return ((mu**2 + 1) ** 0.5 + mu) ** 2

def wellbore_stress_concentration(
    theta: Union[np.ndarray, float],
    sv: Union[np.ndarray, float],
    shmax: Union[np.ndarray, float],
    shmin: Union[np.ndarray, float],
    pp: Union[np.ndarray, float],
    pw: Union[np.ndarray, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate stress concentrations around a wellbore using Kirsch equations.

    Args:
        theta: Azimuth angles around wellbore (radians).
        sv: Vertical stress (MPa).
        shmax: Maximum horizontal stress (MPa).
        shmin: Minimum horizontal stress (MPa).
        pp: Pore pressure (MPa).
        pw: Wellbore pressure (mud pressure, MPa).

    Returns:
        Tuple of (tangential_stress, radial_stress) in effective stress (MPa).

    Example:
        >>> from geosmith.primitives.geomechanics import wellbore_stress_concentration
        >>> import numpy as np
        >>>
        >>> theta = np.linspace(0, 2*np.pi, 360)
        >>> sigma_theta, sigma_r = wellbore_stress_concentration(
        ...     theta, sv=50.0, shmax=45.0, shmin=35.0, pp=20.0, pw=25.0
        ... )
        >>> print(f"Max tangential stress: {sigma_theta.max():.2f} MPa")
    """
    theta = np.asarray(theta, dtype=np.float64)
    sv = np.asarray(sv, dtype=np.float64)
    shmax = np.asarray(shmax, dtype=np.float64)
    shmin = np.asarray(shmin, dtype=np.float64)
    pp = np.asarray(pp, dtype=np.float64)
    pw = np.asarray(pw, dtype=np.float64)

    # Effective stresses
    sv_eff = sv - pp
    shmax_eff = shmax - pp
    shmin_eff = shmin - pp
    pw_eff = pw - pp

    # Average horizontal stress
    sh_avg = (shmax_eff + shmin_eff) / 2

    # Differential horizontal stress
    dsh = (shmax_eff - shmin_eff) / 2

    # Tangential stress (effective) - Kirsch solution
    # σ_θ = sh_avg - 2*dsh*cos(2*theta) + pw_eff
    # Note: Original formula was σ_θ = sh_avg + 2*dsh*cos(2*theta) + pw_eff
    # but standard Kirsch solution uses negative sign for cos(2*theta) term
    sigma_theta = sh_avg - 2 * dsh * np.cos(2 * theta) + pw_eff

    # Radial stress (effective) - equal to effective wellbore pressure at wall
    sigma_r = pw_eff

    # Handle array broadcasting
    if theta.ndim > 0:
        # theta is array, broadcast other arrays if scalar
        if sv_eff.ndim == 0:
            sigma_theta = sigma_theta.ravel()
            sigma_r = np.full_like(sigma_theta, sigma_r)

    return sigma_theta, sigma_r

def stress_polygon_points(
    sv: float,
    pp: float,
    mu: float = 0.6,
) -> list[tuple[float, float]]:
    """Calculate the stress polygon corner points for a given depth.

    Stress polygon shows valid ranges of Shmin and SHmax for different
    faulting regimes (normal, strike-slip, reverse).

    Args:
        sv: Vertical stress (MPa).
        pp: Pore pressure (MPa).
        mu: Friction coefficient, default 0.6.

    Returns:
        List of (Shmin, SHmax) tuples for stress polygon corners.

    Example:
        >>> from geosmith.primitives.geomechanics import stress_polygon_points
        >>>
        >>> corners = stress_polygon_points(sv=100.0, pp=40.0, mu=0.6)
        >>> print(f"Stress polygon has {len(corners)} corner points")
        >>> for i, (shmin, shmax) in enumerate(corners):
        ...     print(f"  Corner {i+1}: Shmin={shmin:.1f}, SHmax={shmax:.1f}")
    """
    f_mu = friction_coefficient_ratio(mu)

    # Calculate limiting stress values
    s1_max = f_mu * (sv - pp) + pp  # Maximum S1 when Sv = S3
    s3_min = ((sv - pp) / f_mu) + pp  # Minimum S3 when Sv = S1

    # Stress polygon corners (moving from bottom-left to top-right)
    corners = [
        (s3_min, s3_min),  # Corner 1: Normal faulting lower bound
        (s3_min, sv),  # Corner 2: Transition to strike-slip
        (sv, sv),  # Corner 3: Strike-slip center
        (sv, s1_max),  # Corner 4: Transition to reverse
        (s1_max, s1_max),  # Corner 5: Reverse faulting upper bound
    ]

    return corners

def shmax_bounds(
    sv: float,
    shmin: float,
    pp: float,
    mu: float = 0.6,
) -> dict[str, float]:
    """Calculate SHmax bounds for different faulting regimes.

    Args:
        sv: Vertical stress (MPa).
        shmin: Minimum horizontal stress (MPa).
        pp: Pore pressure (MPa).
        mu: Friction coefficient, default 0.6.

    Returns:
        Dictionary with SHmax bounds for each faulting regime:
            - 'shmax_strike_slip': SHmax for strike-slip faulting
            - 'shmax_reverse': SHmax for reverse faulting
            - 'shmin_normal': Shmin for normal faulting (reference)
            - 'shmax_tensile_fracture': SHmax for tensile fracture (upper bound)
            - 'friction_ratio': Friction coefficient ratio

    Example:
        >>> from geosmith.primitives.geomechanics import shmax_bounds
        >>>
        >>> bounds = shmax_bounds(sv=100.0, shmin=60.0, pp=40.0, mu=0.6)
        >>> print(f"SHmax for strike-slip: {bounds['shmax_strike_slip']:.1f} MPa")
        >>> print(f"SHmax for reverse: {bounds['shmax_reverse']:.1f} MPa")
    """
    f_mu = friction_coefficient_ratio(mu)

    # Strike-slip faulting: SHmax = fμ * (Shmin - Pp) + Pp
    shmax_ss = f_mu * (shmin - pp) + pp

    # Reverse faulting: SHmax = fμ * (Sv - Pp) + Pp
    shmax_rev = f_mu * (sv - pp) + pp

    # Normal faulting: Shmin = ((Sv - Pp) / fμ) + Pp (for reference)
    shmin_nf = ((sv - pp) / f_mu) + pp

    # Tensile fracture constraint: SHmax_tf = 3*Shmin - 2*Pp (simplified)
    shmax_tf = 3 * shmin - 2 * pp

    return {
        "shmax_strike_slip": shmax_ss,
        "shmax_reverse": shmax_rev,
        "shmin_normal": shmin_nf,
        "shmax_tensile_fracture": shmax_tf,
        "friction_ratio": f_mu,
    }

def determine_stress_regime(
    sv: Union[np.ndarray, float],
    shmin: Union[np.ndarray, float],
    shmax: Union[np.ndarray, float],
    tolerance: float = 0.1,
) -> Union[np.ndarray, str]:
    """Determine stress regime from principal stresses.

    Classifies faulting regime based on relative magnitudes of principal stresses.

    Args:
        sv: Vertical stress (MPa).
        shmin: Minimum horizontal stress (MPa).
        shmax: Maximum horizontal stress (MPa).
        tolerance: Tolerance for numerical comparison (MPa), default 0.1.

    Returns:
        Stress regime string or array: 'normal', 'strike_slip', 'reverse', or 'unknown'.

    Example:
        >>> from geosmith.primitives.geomechanics import determine_stress_regime
        >>> import numpy as np
        >>>
        >>> sv = np.array([100.0, 110.0, 120.0])
        >>> shmin = np.array([60.0, 70.0, 80.0])
        >>> shmax = np.array([90.0, 100.0, 110.0])
        >>> regimes = determine_stress_regime(sv, shmin, shmax)
        >>> print(regimes)
    """
    sv = np.asarray(sv, dtype=np.float64)
    shmin = np.asarray(shmin, dtype=np.float64)
    shmax = np.asarray(shmax, dtype=np.float64)

    if sv.shape != shmin.shape or sv.shape != shmax.shape:
        raise ValueError("sv, shmin, and shmax must have the same shape")

    # For scalar inputs
    if sv.ndim == 0:
        s1, s2, s3 = sorted([float(sv), float(shmin), float(shmax)], reverse=True)

        # Map vertical stress position to faulting regime
        if np.isclose(sv, s1, atol=tolerance):
            return "normal"  # Sv ≈ S1 (largest)
        elif np.isclose(sv, s3, atol=tolerance):
            return "reverse"  # Sv ≈ S3 (smallest)
        elif np.isclose(sv, s2, atol=tolerance):
            return "strike_slip"  # Sv ≈ S2 (intermediate)
        else:
            return "unknown"

    # For array inputs
    regimes = np.full_like(sv, "unknown", dtype=object)

    for i in range(len(sv)):
        s1, s2, s3 = sorted([sv[i], shmin[i], shmax[i]], reverse=True)

        if np.isclose(sv[i], s1, atol=tolerance):
            regimes[i] = "normal"
        elif np.isclose(sv[i], s3, atol=tolerance):
            regimes[i] = "reverse"
        elif np.isclose(sv[i], s2, atol=tolerance):
            regimes[i] = "strike_slip"

    return regimes

