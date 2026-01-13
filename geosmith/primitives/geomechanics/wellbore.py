"""Geomechanics: Wellbore stability analysis

Pure geomechanics operations - wellbore module.
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

from dataclasses import dataclass

from geosmith.primitives.geomechanics.pressure import (
    calculate_hydrostatic_pressure,
    pore_pressure_eaton,
    sv_from_density,
)
from geosmith.primitives.geomechanics.stress import (
    shmax_bounds,
    wellbore_stress_concentration,
)
from geosmith.primitives.petrophysics import calculate_porosity_from_density


@dataclass
class GeomechParams:
    """Parameters for geomechanical calculations.

    Attributes:
        phi_initial: Initial porosity (fraction), default 0.35.
        beta: Compaction coefficient (1/MPa), default 0.03.
        mu: Friction coefficient (dimensionless), default 0.69.
        ucs: Unconfined Compressive Strength (MPa), default 30.0.
        tensile_strength: Rock tensile strength (MPa), default 0.0.
        delta_p: Excess pressure from drilling (MPa), default 0.0.
    """

    phi_initial: float = 0.35
    beta: float = 0.03
    mu: float = 0.69
    ucs: float = 30.0
    tensile_strength: float = 0.0
    delta_p: float = 0.0


@dataclass
class WellboreStabilityResult:
    """Results from wellbore stability analysis.

    Attributes:
        breakout_pressure: Minimum mud pressure to prevent breakout (MPa).
        fracture_pressure: Maximum mud pressure before tensile failure (MPa).
        breakout_width: Angular width of breakout (degrees).
        breakout_azimuth: Azimuth of breakout (degrees from SHmax).
        safe_mud_weight_min: Minimum safe mud weight (g/cc).
        safe_mud_weight_max: Maximum safe mud weight (g/cc).
        stability_margin: Stability margin (fraction).
    """

    breakout_pressure: float
    fracture_pressure: float
    breakout_width: float
    breakout_azimuth: float
    safe_mud_weight_min: float
    safe_mud_weight_max: float
    stability_margin: float


def breakout_analysis(
    sv: float,
    shmax: float,
    shmin: float,
    pp: float,
    params: GeomechParams,
    depth: float = 1000.0,
) -> WellboreStabilityResult:
    """Perform comprehensive wellbore stability analysis.

    Uses Mohr-Coulomb failure criterion to determine safe mud weight window.

    Args:
        sv: Vertical stress (MPa).
        shmax: Maximum horizontal stress (MPa).
        shmin: Minimum horizontal stress (MPa).
        pp: Pore pressure (MPa).
        params: GeomechParams with UCS, tensile strength, etc.
        depth: Depth for mud weight calculations (m), default 1000.0.

    Returns:
        WellboreStabilityResult with all stability parameters.

    Example:
        >>> from geosmith.primitives.geomechanics import (
        ...     breakout_analysis, GeomechParams
        ... )
        >>>
        >>> params = GeomechParams(ucs=50.0, mu=0.6, tensile_strength=5.0)
        >>> result = breakout_analysis(
        ...     sv=100.0, shmax=80.0, shmin=60.0, pp=40.0,
        ...     params=params, depth=2000.0
        ... )
        >>> print(
        ...     f"Safe mud weight window: "
        ...     f"{result.safe_mud_weight_min:.2f} - "
        ...     f"{result.safe_mud_weight_max:.2f} g/cc"
        ... )
    """
    # Breakout analysis - find minimum mud pressure to prevent failure
    # Using Mohr-Coulomb failure criterion at wellbore wall

    # Critical angle for maximum tangential stress (90° from SHmax)
    theta_critical = np.pi / 2  # 90 degrees

    # Convert friction coefficient to friction angle
    phi = np.arctan(params.mu)  # friction angle
    cohesion = params.ucs * (1 - np.sin(phi)) / (2 * np.cos(phi))

    # Effective stresses
    sv_eff = sv - pp
    shmax_eff = shmax - pp
    shmin_eff = shmin - pp

    # Average and differential horizontal stress
    sh_avg_eff = (shmax_eff + shmin_eff) / 2
    dsh_eff = (shmax_eff - shmin_eff) / 2

    # At breakout location (θ = π/2), tangential stress is maximum
    # σ_θ = sh_avg - 2*dsh + pw_eff (at θ = π/2)
    # For stability: σ_θ_eff ≤ UCS_eff = UCS (since UCS is already effective stress)

    # Minimum effective wellbore pressure to prevent breakout
    pw_eff_min = params.ucs - sh_avg_eff + 2 * dsh_eff
    breakout_pressure = pw_eff_min + pp

    # Tensile fracture analysis - maximum mud pressure
    # Tensile failure occurs when tangential stress becomes tensile
    # At minimum stress location (θ = 0), σ_θ = sh_avg + 2*dsh + pw_eff
    # For tensile failure: σ_θ ≤ -T0 (tensile strength is negative)

    # Maximum effective wellbore pressure before tensile failure
    pw_eff_max = -params.tensile_strength - sh_avg_eff - 2 * dsh_eff
    fracture_pressure = pw_eff_max + pp

    # Alternative fracture pressure using LOT relationship
    # SHmax_tf = 3*Shmin - 2*Pp - ΔP
    fracture_pressure_alt = 3 * shmin - 2 * pp - params.delta_p
    fracture_pressure = min(fracture_pressure, fracture_pressure_alt)

    # Breakout width calculation
    # Simplified model: breakout occurs where tangential stress exceeds UCS
    theta_range = np.linspace(0, 2 * np.pi, 360)
    sigma_theta, _ = wellbore_stress_concentration(
        theta_range, sv, shmax, shmin, pp, breakout_pressure
    )

    # Find angles where failure occurs (effective tangential stress > UCS)
    failed_indices = sigma_theta > params.ucs
    if np.any(failed_indices):
        failed_angles = theta_range[failed_indices]
        # Breakout typically occurs in two symmetric lobes
        if len(failed_angles) > 0:
            breakout_width = np.degrees(
                np.ptp(failed_angles[failed_angles < np.pi])
            )  # Width of one lobe
        else:
            breakout_width = 0.0
    else:
        breakout_width = 0.0

    # Breakout azimuth (perpendicular to SHmax direction)
    breakout_azimuth = 90.0  # degrees from SHmax

    # Convert to mud weights
    safe_mud_weight_min = breakout_pressure / (depth * 0.00981) if depth > 0 else 0
    safe_mud_weight_max = fracture_pressure / (depth * 0.00981) if depth > 0 else 0

    # Stability margin
    pressure_window = fracture_pressure - breakout_pressure
    stability_margin = pressure_window / max(breakout_pressure, 1e-6)

    return WellboreStabilityResult(
        breakout_pressure=breakout_pressure,
        fracture_pressure=fracture_pressure,
        breakout_width=max(0, breakout_width),
        breakout_azimuth=breakout_azimuth,
        safe_mud_weight_min=max(0, safe_mud_weight_min),
        safe_mud_weight_max=max(0, safe_mud_weight_max),
        stability_margin=stability_margin,
    )


def safe_mud_weight_window(
    df: "pd.DataFrame", params: GeomechParams, shmin_estimate: Optional[float] = None
) -> "pd.DataFrame":
    """Calculate safe mud weight window for entire well trajectory.

    Args:
        df: DataFrame with 'depth_m' and 'RHOB' columns (depth in meters,
            density in g/cc).
        params: GeomechParams with geomechanical parameters.
        shmin_estimate: Optional estimate of Shmin (MPa). If None, uses stress polygon.

    Returns:
        DataFrame with safe mud weight analysis vs depth, including:
        - depth_m: Depth (m)
        - sv_mpa: Vertical stress (MPa)
        - shmax_mpa: Maximum horizontal stress (MPa)
        - shmin_mpa: Minimum horizontal stress (MPa)
        - pp_mpa: Pore pressure (MPa)
        - breakout_pressure_mpa: Minimum pressure to prevent breakout (MPa)
        - fracture_pressure_mpa: Maximum pressure before fracture (MPa)
        - mud_weight_min_gcc: Minimum safe mud weight (g/cc)
        - mud_weight_max_gcc: Maximum safe mud weight (g/cc)
        - breakout_width_deg: Breakout width (degrees)
        - stability_margin: Stability margin (fraction)

    Example:
        >>> from geosmith.primitives.geomechanics import (
        ...     safe_mud_weight_window, GeomechParams
        ... )
        >>> import pandas as pd
        >>>
        >>> df = pd.DataFrame({'depth_m': [1000, 2000, 3000], 'RHOB': [2.2, 2.4, 2.6]})
        >>> params = GeomechParams()
        >>> result = safe_mud_weight_window(df, params)
        >>> print(
        ...     f"Mud weight window: "
        ...     f"{result['mud_weight_min_gcc'].min():.2f} - "
        ...     f"{result['mud_weight_max_gcc'].max():.2f} g/cc"
        ... )
    """
    if not PANDAS_AVAILABLE:
        raise ImportError(
            "pandas is required for safe_mud_weight_window. "
            "Install with: pip install pandas"
        )

    from geosmith.primitives.petrophysics import calculate_porosity_from_density

    results = []

    # Calculate porosity and stresses
    from geosmith.primitives.petrophysics import calculate_porosity_from_density

    porosity = calculate_porosity_from_density(df["RHOB"], matrix_density=2.65)
    sv = sv_from_density(df["depth_m"], df["RHOB"])
    ph = calculate_hydrostatic_pressure(df["depth_m"])
    pp = pore_pressure_eaton(sv, porosity, params.phi_initial, params.beta)

    for i, row in df.iterrows():
        depth = row["depth_m"]
        sv_val = sv.iloc[i] if hasattr(sv, "iloc") else sv[i]
        pp_val = pp.iloc[i] if hasattr(pp, "iloc") else pp[i]

        # Estimate horizontal stresses if not provided
        if shmin_estimate is None:
            # Use simple relationship: Shmin ≈ 0.7 * Sv (typical for normal faulting)
            shmin_val = 0.7 * sv_val
        else:
            shmin_val = shmin_estimate

        # Estimate SHmax using stress polygon bounds
        bounds = shmax_bounds(sv_val, shmin_val, pp_val, params.mu)
        shmax_val = min(bounds["shmax_strike_slip"], bounds["shmax_tensile_fracture"])

        # Perform wellbore stability analysis
        stability = breakout_analysis(
            sv_val, shmax_val, shmin_val, pp_val, params, depth
        )

        results.append(
            {
                "depth_m": depth,
                "sv_mpa": sv_val,
                "shmax_mpa": shmax_val,
                "shmin_mpa": shmin_val,
                "pp_mpa": pp_val,
                "breakout_pressure_mpa": stability.breakout_pressure,
                "fracture_pressure_mpa": stability.fracture_pressure,
                "mud_weight_min_gcc": stability.safe_mud_weight_min,
                "mud_weight_max_gcc": stability.safe_mud_weight_max,
                "breakout_width_deg": stability.breakout_width,
                "stability_margin": stability.stability_margin,
            }
        )

    return pd.DataFrame(results)


def wellbore_stress_plot_data(
    sv: float,
    shmax: float,
    shmin: float,
    pp: float,
    pw: float,
    num_points: int = 360,
) -> "pd.DataFrame":
    """Generate data for wellbore stress concentration plot.

    Args:
        sv: Vertical stress (MPa).
        shmax: Maximum horizontal stress (MPa).
        shmin: Minimum horizontal stress (MPa).
        pp: Pore pressure (MPa).
        pw: Wellbore pressure (mud pressure, MPa).
        num_points: Number of points around wellbore circumference, default 360.

    Returns:
        DataFrame with columns:
        - theta_deg: Azimuth angle (degrees)
        - theta_rad: Azimuth angle (radians)
        - sigma_theta_eff: Tangential stress (effective, MPa)
        - sigma_r_eff: Radial stress (effective, MPa)
        - sigma_theta_total: Tangential stress (total, MPa)
        - sigma_r_total: Radial stress (total, MPa)

    Example:
        >>> from geosmith.primitives.geomechanics import wellbore_stress_plot_data
        >>>
        >>> plot_data = wellbore_stress_plot_data(
        ...     sv=100.0, shmax=80.0, shmin=60.0, pp=40.0, pw=50.0
        ... )
        >>> print(
        ...     f"Max tangential stress: "
        ...     f"{plot_data['sigma_theta_eff'].max():.1f} MPa"
        ... )
    """
    if not PANDAS_AVAILABLE:
        raise ImportError(
            "pandas is required for wellbore_stress_plot_data. "
            "Install with: pip install pandas"
        )

    theta = np.linspace(0, 2 * np.pi, num_points)
    sigma_theta, sigma_r = wellbore_stress_concentration(
        theta, sv, shmax, shmin, pp, pw
    )

    return pd.DataFrame(
        {
            "theta_deg": np.degrees(theta),
            "theta_rad": theta,
            "sigma_theta_eff": sigma_theta,
            "sigma_r_eff": sigma_r,
            "sigma_theta_total": sigma_theta + pp,
            "sigma_r_total": sigma_r + pp,
        }
    )


def drilling_margin_analysis(
    breakout_pressure: float,
    fracture_pressure: float,
    current_mud_weight: float,
    depth: float,
) -> Dict[str, float]:
    """Analyze drilling safety margins for current mud weight.

    Args:
        breakout_pressure: Minimum pressure to prevent breakout (MPa).
        fracture_pressure: Maximum pressure before fracture (MPa).
        current_mud_weight: Current mud weight (g/cc).
        depth: Depth (m).

    Returns:
        Dictionary with margin analysis:
        - current_pressure_mpa: Current wellbore pressure (MPa)
        - breakout_margin_pct: Margin above breakout pressure (%)
        - fracture_margin_pct: Margin below fracture pressure (%)
        - pressure_window_mpa: Safe pressure window size (MPa)
        - safety_status: 'SAFE', 'BREAKOUT RISK', or 'FRACTURE RISK'
        - recommended_mud_weight_gcc: Recommended mud weight (g/cc)

    Example:
        >>> from geosmith.primitives.geomechanics import drilling_margin_analysis
        >>>
        >>> margin = drilling_margin_analysis(
        ...     breakout_pressure=30.0, fracture_pressure=50.0,
        ...     current_mud_weight=1.2, depth=2000.0
        ... )
        >>> print(f"Safety status: {margin['safety_status']}")
    """
    current_pressure = current_mud_weight * depth * 0.00981

    # Calculate margins
    breakout_margin = (
        (current_pressure - breakout_pressure) / breakout_pressure * 100
        if breakout_pressure > 0
        else 0.0
    )
    fracture_margin = (
        (fracture_pressure - current_pressure) / fracture_pressure * 100
        if fracture_pressure > 0
        else 0.0
    )

    # Safety status
    if current_pressure < breakout_pressure:
        status = "BREAKOUT RISK"
    elif current_pressure > fracture_pressure:
        status = "FRACTURE RISK"
    else:
        status = "SAFE"

    return {
        "current_pressure_mpa": current_pressure,
        "breakout_margin_pct": breakout_margin,
        "fracture_margin_pct": fracture_margin,
        "pressure_window_mpa": fracture_pressure - breakout_pressure,
        "safety_status": status,
        "recommended_mud_weight_gcc": (
            (breakout_pressure + fracture_pressure) / 2 / (depth * 0.00981)
            if depth > 0
            else 0.0
        ),
    }
