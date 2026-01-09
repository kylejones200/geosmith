"""Geomechanics calculation primitives.

Pure geomechanics operations.
Migrated from geosuite.geomech.stresses.
Layer 2: Primitives - Pure operations.
"""

from typing import Optional, Union

import numpy as np

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator if not args else decorator(args[0])


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


def calculate_overpressure(
    pp: Union[np.ndarray, float],
    ph: Union[np.ndarray, float],
) -> np.ndarray:
    """Calculate overpressure.

    ΔP = Pp - Ph

    Args:
        pp: Pore pressure (MPa).
        ph: Hydrostatic pressure (MPa).

    Returns:
        Overpressure (MPa).

    Example:
        >>> from geosmith.primitives.geomechanics import calculate_overpressure
        >>>
        >>> overpressure = calculate_overpressure(pp=25.0, ph=20.0)
        >>> print(f"Overpressure: {overpressure:.1f} MPa")
    """
    return np.asarray(pp, dtype=float) - np.asarray(ph, dtype=float)


@njit(cache=True)
def _calculate_pressure_gradient_kernel(
    pressure: np.ndarray, depth: np.ndarray
) -> np.ndarray:
    """Numba-optimized kernel for pressure gradient calculation.

    Args:
        pressure: Pressure array (MPa).
        depth: Depth array (meters).

    Returns:
        Pressure gradient (MPa/m).
    """
    n = len(pressure)
    gradient = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        dz = depth[i] - depth[i - 1]
        if dz > 0.0:
            gradient[i] = (pressure[i] - pressure[i - 1]) / dz
        else:
            gradient[i] = gradient[i - 1] if i > 1 else 0.0

    # Extrapolate first value
    gradient[0] = gradient[1] if n > 1 else 0.0

    return gradient


def calculate_pressure_gradient(
    pressure: Union[np.ndarray, float],
    depth: Union[np.ndarray, float],
) -> np.ndarray:
    """Calculate pressure gradient (MPa/m or equivalent mud weight).

    Args:
        pressure: Pressure array (MPa).
        depth: Depth array (meters).

    Returns:
        Pressure gradient (MPa/m) as numpy array.

    Example:
        >>> from geosmith.primitives.geomechanics import calculate_pressure_gradient
        >>>
        >>> pressure = np.array([10, 20, 30])
        >>> depth = np.array([1000, 2000, 3000])
        >>> gradient = calculate_pressure_gradient(pressure, depth)
        >>> print(f"Gradient: {gradient.mean():.4f} MPa/m")
    """
    pressure = np.asarray(pressure, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)

    if len(pressure) == 0 or len(depth) == 0:
        raise ValueError("Pressure and depth arrays must not be empty")

    if len(pressure) != len(depth):
        raise ValueError("Pressure and depth arrays must have same length")

    # Call optimized kernel
    return _calculate_pressure_gradient_kernel(pressure, depth)


def pressure_to_mud_weight(
    pressure: Union[np.ndarray, float],
    depth: Union[np.ndarray, float],
    g: float = 9.81,
) -> Union[np.ndarray, float]:
    """Convert pressure to equivalent mud weight.

    MW = Pressure / (g * depth)

    Args:
        pressure: Pressure (MPa).
        depth: Depth (meters).
        g: Gravitational acceleration (m/s²), default 9.81.

    Returns:
        Mud weight (g/cc) as numpy array or float.

    Example:
        >>> from geosmith.primitives.geomechanics import pressure_to_mud_weight
        >>>
        >>> mw = pressure_to_mud_weight(pressure=20.0, depth=2000.0)
        >>> print(f"Mud weight: {mw:.2f} g/cc")
    """
    pressure = np.asarray(pressure, dtype=float)
    depth = np.asarray(depth, dtype=float)

    # Handle scalar inputs
    is_scalar = pressure.ndim == 0 and depth.ndim == 0

    if pressure.size == 0 or depth.size == 0:
        raise ValueError("Pressure and depth arrays must not be empty")

    if pressure.size != depth.size and not is_scalar:
        raise ValueError("Pressure and depth arrays must have same length")

    # Avoid division by zero
    depth = np.where(depth <= 0, np.nan, depth)

    # Convert MPa to Pa, calculate density in kg/m³, then convert to g/cc
    mw = (pressure * 1e6) / (g * depth) / 1000

    # Return scalar if input was scalar
    if is_scalar:
        return float(mw)
    return mw


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


def mohr_coulomb_failure(
    sigma1: Union[np.ndarray, float],
    sigma3: Union[np.ndarray, float],
    cohesion: float = 10.0,
    friction_angle: float = 30.0,
) -> tuple[Union[np.ndarray, float], Union[np.ndarray, float]]:
    """Calculate Mohr-Coulomb failure criterion.

    sigma1_fail = sigma3 * tan^2(45 + phi/2) + 2 * c * tan(45 + phi/2)

    Args:
        sigma1: Maximum principal stress (MPa).
        sigma3: Minimum principal stress (MPa).
        cohesion: Cohesion (MPa).
        friction_angle: Friction angle in degrees.

    Returns:
        Tuple of (failure_stress, safety_factor).

    Example:
        >>> from geosmith.primitives.geomechanics import mohr_coulomb_failure
        >>>
        >>> sigma1_fail, safety = mohr_coulomb_failure(
        ...     sigma1=50.0, sigma3=20.0, cohesion=10.0, friction_angle=30.0
        ... )
        >>> print(f"Failure stress: {sigma1_fail:.1f} MPa, Safety factor: {safety:.2f}")
    """
    sigma1 = np.asarray(sigma1, dtype=float)
    sigma3 = np.asarray(sigma3, dtype=float)

    phi_rad = np.radians(friction_angle)
    tan_squared = np.tan(np.radians(45.0 + friction_angle / 2.0)) ** 2

    # Failure stress
    sigma1_fail = sigma3 * tan_squared + 2 * cohesion * np.sqrt(tan_squared)

    # Safety factor
    safety_factor = sigma1_fail / sigma1

    return sigma1_fail, safety_factor


def drucker_prager_failure(
    sigma1: Union[np.ndarray, float],
    sigma2: Union[np.ndarray, float],
    sigma3: Union[np.ndarray, float],
    cohesion: float = 10.0,
    friction_angle: float = 30.0,
) -> tuple[Union[np.ndarray, float], Union[np.ndarray, float]]:
    """Calculate Drucker-Prager failure criterion.

    Uses mean stress and deviatoric stress invariants.

    Args:
        sigma1: Maximum principal stress (MPa).
        sigma2: Intermediate principal stress (MPa).
        sigma3: Minimum principal stress (MPa).
        cohesion: Cohesion (MPa).
        friction_angle: Friction angle in degrees.

    Returns:
        Tuple of (failure_stress, safety_factor).

    Example:
        >>> from geosmith.primitives.geomechanics import drucker_prager_failure
        >>>
        >>> sqrt_J2_fail, safety = drucker_prager_failure(
        ...     sigma1=50.0, sigma2=35.0, sigma3=20.0,
        ...     cohesion=10.0, friction_angle=30.0
        ... )
    """
    sigma1 = np.asarray(sigma1, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)
    sigma3 = np.asarray(sigma3, dtype=float)

    phi_rad = np.radians(friction_angle)

    # Mean stress
    I1 = sigma1 + sigma2 + sigma3

    # Deviatoric stress
    J2 = (
        (sigma1 - sigma2) ** 2
        + (sigma2 - sigma3) ** 2
        + (sigma3 - sigma1) ** 2
    ) / 6.0

    # Drucker-Prager parameters
    alpha = np.sin(phi_rad) / np.sqrt(3.0 * (3.0 + np.sin(phi_rad) ** 2))
    k = (
        np.sqrt(3.0)
        * cohesion
        * np.cos(phi_rad)
        / np.sqrt(3.0 + np.sin(phi_rad) ** 2)
    )

    # Failure criterion: sqrt(J2) = alpha * I1 + k
    sqrt_J2_fail = alpha * I1 + k
    sqrt_J2_actual = np.sqrt(J2)

    # Safety factor
    safety_factor = sqrt_J2_fail / sqrt_J2_actual

    return sqrt_J2_fail, safety_factor


def hoek_brown_failure(
    sigma1: Union[np.ndarray, float],
    sigma3: Union[np.ndarray, float],
    ucs: float = 50.0,
    mi: float = 15.0,
    gsi: float = 75.0,
    d: float = 0.0,
) -> tuple[Union[np.ndarray, float], Union[np.ndarray, float]]:
    """Calculate Hoek-Brown failure criterion for rock masses.

    sigma1 = sigma3 + ucs * (mb * sigma3 / ucs + s)^a

    Args:
        sigma1: Maximum principal stress (MPa).
        sigma3: Minimum principal stress (MPa).
        ucs: Unconfined compressive strength (MPa).
        mi: Intact rock parameter.
        gsi: Geological Strength Index (0-100).
        d: Disturbance factor (0-1).

    Returns:
        Tuple of (failure_stress, safety_factor).

    Example:
        >>> from geosmith.primitives.geomechanics import hoek_brown_failure
        >>>
        >>> sigma1_fail, safety = hoek_brown_failure(
        ...     sigma1=50.0, sigma3=20.0, ucs=50.0, mi=15.0, gsi=75.0
        ... )
    """
    sigma1 = np.asarray(sigma1, dtype=float)
    sigma3 = np.asarray(sigma3, dtype=float)

    # Hoek-Brown parameters
    mb = mi * np.exp((gsi - 100) / (28 - 14 * d))
    s = np.exp((gsi - 100) / (9 - 3 * d))
    a = 0.5 + (1.0 / 6.0) * (np.exp(-gsi / 15.0) - np.exp(-20.0 / 3.0))

    # Failure stress
    sigma1_fail = sigma3 + ucs * (mb * sigma3 / ucs + s) ** a

    # Safety factor
    safety_factor = sigma1_fail / sigma1

    return sigma1_fail, safety_factor



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
    # SHmax_min = q * (Sv - Pp) + C + Pp
    # If Shmin is known: SHmax_max = q * (Shmin - Pp) + C + Pp
    rf_min = q * sv_eff + cohesion + pp
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


@njit(cache=True)
def _calculate_overburden_stress_kernel(
    depth: np.ndarray, rhob_kg: np.ndarray, g: float
) -> np.ndarray:
    """Numba-optimized kernel for overburden stress integration.

    This function is JIT-compiled for 20-50x speedup on large datasets.

    Args:
        depth: Depth array (meters).
        rhob_kg: Bulk density array (kg/m³).
        g: Gravitational acceleration (m/s²).

    Returns:
        Overburden stress (MPa).
    """
    n = len(depth)
    sv = np.zeros(n, dtype=np.float64)

    # Trapezoidal integration: accumulate density × gravity × depth increment
    for i in range(1, n):
        dz = depth[i] - depth[i - 1]
        if dz > 0.0:
            avg_rho = (rhob_kg[i] + rhob_kg[i - 1]) * 0.5
            sv[i] = sv[i - 1] + avg_rho * g * dz * 1e-6  # Convert Pa to MPa
        else:
            sv[i] = sv[i - 1] if i > 1 else 0.0

    return sv


def calculate_overburden_stress(
    depth: Union[np.ndarray, float],
    rhob: Union[np.ndarray, float],
    g: float = 9.81,
) -> np.ndarray:
    """Calculate overburden stress (Sv) from density log.

    Uses trapezoidal integration: Sv = integral(rho * g * dz)

    This function is accelerated with Numba JIT compilation for 20-50x speedup
    on datasets with 1000+ samples. Falls back to pure Python if Numba unavailable.

    Args:
        depth: Depth array (meters).
        rhob: Bulk density array (g/cc).
        g: Gravitational acceleration (m/s²), default 9.81.

    Returns:
        Overburden stress (MPa) as numpy array.

    Example:
        >>> from geosmith.primitives.geomechanics import calculate_overburden_stress
        >>>
        >>> depth = np.linspace(0, 3000, 1000)  # 0-3000m
        >>> rhob = np.ones(1000) * 2.5  # 2.5 g/cc
        >>> sv = calculate_overburden_stress(depth, rhob)
        >>> print(f"Overburden at {depth[-1]}m: {sv[-1]:.1f} MPa")
    """
    depth = np.asarray(depth, dtype=np.float64)
    rhob = np.asarray(rhob, dtype=np.float64)

    if len(depth) == 0 or len(rhob) == 0:
        raise ValueError("Depth and density arrays must not be empty")

    if len(depth) != len(rhob):
        raise ValueError("Depth and density arrays must have same length")

    # Convert g/cc to kg/m³
    rhob_kg = rhob * 1000.0

    # Call optimized kernel
    if NUMBA_AVAILABLE:
        return _calculate_overburden_stress_kernel(depth, rhob_kg, g)
    else:
        # Fallback to pure Python
        n = len(depth)
        sv = np.zeros(n, dtype=np.float64)

        for i in range(1, n):
            dz = depth[i] - depth[i - 1]
            if dz > 0.0:
                avg_rho = (rhob_kg[i] + rhob_kg[i - 1]) * 0.5
                sv[i] = sv[i - 1] + avg_rho * g * dz * 1e-6
            else:
                sv[i] = sv[i - 1] if i > 1 else 0.0

        return sv


def calculate_hydrostatic_pressure(
    depth: Union[np.ndarray, float],
    rho_water: float = 1.03,
    g: float = 9.81,
) -> np.ndarray:
    """Calculate hydrostatic pressure.

    Ph = rho_water * g * depth

    Args:
        depth: Depth array (meters).
        rho_water: Water density (g/cc), default 1.03.
        g: Gravitational acceleration (m/s²), default 9.81.

    Returns:
        Hydrostatic pressure (MPa) as numpy array.

    Example:
        >>> from geosmith.primitives.geomechanics import calculate_hydrostatic_pressure
        >>>
        >>> depth = np.array([1000, 2000, 3000])
        >>> ph = calculate_hydrostatic_pressure(depth, rho_water=1.03)
        >>> print(f"Hydrostatic pressure at {depth[-1]}m: {ph[-1]:.1f} MPa")
    """
    depth = np.asarray(depth, dtype=np.float64)

    if len(depth) == 0:
        raise ValueError("Depth array must not be empty")

    rho_water_kg = rho_water * 1000.0  # Convert to kg/m³
    ph = rho_water_kg * g * depth / 1e6  # Convert Pa to MPa

    return ph



def predict_fracture_orientation(
    shmax_azimuth: Union[np.ndarray, float],
    shmin_azimuth: Union[np.ndarray, float],
    stress_ratio: Union[np.ndarray, float],
    method: str = "coulomb",
) -> dict[str, Union[np.ndarray, float]]:
    """Predict fracture orientation from stress field.

    Natural fractures typically form perpendicular to Shmin.
    Induced fractures form in direction of maximum stress.

    Args:
        shmax_azimuth: Maximum horizontal stress azimuth (degrees from north).
        shmin_azimuth: Minimum horizontal stress azimuth (degrees from north).
        stress_ratio: Ratio of (SHmax - Pp) / (Sv - Pp).
        method: Prediction method ('coulomb', 'griffith', 'tensile').

    Returns:
        Dictionary with fracture strike, dip, type, and azimuth.

    Example:
        >>> from geosmith.primitives.geomechanics import predict_fracture_orientation
        >>>
        >>> result = predict_fracture_orientation(
        ...     shmax_azimuth=45.0, shmin_azimuth=135.0, stress_ratio=1.2
        ... )
        >>> print(f"Fracture strike: {result['strike']:.1f}°")
    """
    shmax_azimuth = np.asarray(shmax_azimuth, dtype=float)
    shmin_azimuth = np.asarray(shmin_azimuth, dtype=float)
    stress_ratio = np.asarray(stress_ratio, dtype=float)

    if method == "coulomb":
        return _predict_coulomb_fracture(shmax_azimuth, shmin_azimuth, stress_ratio)
    elif method == "griffith":
        return _predict_griffith_fracture(shmax_azimuth, shmin_azimuth, stress_ratio)
    elif method == "tensile":
        return _predict_tensile_fracture(shmax_azimuth, shmin_azimuth, stress_ratio)
    else:
        raise ValueError(
            f"Unknown method: {method}. Choose: 'coulomb', 'griffith', 'tensile'"
        )


def _predict_coulomb_fracture(
    shmax_azimuth: np.ndarray,
    shmin_azimuth: np.ndarray,
    stress_ratio: np.ndarray,
) -> dict[str, np.ndarray]:
    """Predict fracture orientation using Coulomb failure criterion."""
    # Natural fractures form at angle to Shmax based on friction
    friction_angle = 30.0  # degrees
    fracture_angle = 45.0 - friction_angle / 2.0

    # Fracture strike is perpendicular to Shmin for normal faulting
    # For strike-slip, fractures form at angle to Shmax
    is_normal = stress_ratio < 1.0
    is_strike_slip = (stress_ratio >= 1.0) & (stress_ratio < 1.5)

    fracture_strike = np.where(
        is_normal,
        shmin_azimuth + 90.0,  # Perpendicular to Shmin
        np.where(
            is_strike_slip,
            shmax_azimuth + fracture_angle,  # At angle to Shmax
            shmax_azimuth,  # Parallel to Shmax for reverse
        ),
    ) % 360.0

    # Fracture dip depends on stress regime
    fracture_dip = np.where(
        is_normal,
        60.0,  # Steep dip for normal faulting
        np.where(is_strike_slip, 90.0, 30.0),  # Vertical for strike-slip, shallow for reverse
    )

    fracture_type = np.where(
        is_normal, "normal", np.where(is_strike_slip, "strike_slip", "reverse")
    )

    return {
        "strike": fracture_strike,
        "dip": fracture_dip,
        "type": fracture_type,
        "azimuth": fracture_strike,
    }


def _predict_griffith_fracture(
    shmax_azimuth: np.ndarray,
    shmin_azimuth: np.ndarray,
    stress_ratio: np.ndarray,
) -> dict[str, np.ndarray]:
    """Predict fracture orientation using Griffith failure criterion."""
    # Griffith theory: fractures form when tensile stress exceeds strength
    # Typically forms perpendicular to minimum principal stress
    fracture_strike = (shmin_azimuth + 90.0) % 360.0
    fracture_dip = np.full_like(fracture_strike, 90.0)  # Vertical fractures
    fracture_type = np.full(len(fracture_strike), "tensile", dtype=object)

    return {
        "strike": fracture_strike,
        "dip": fracture_dip,
        "type": fracture_type,
        "azimuth": fracture_strike,
    }


def _predict_tensile_fracture(
    shmax_azimuth: np.ndarray,
    shmin_azimuth: np.ndarray,
    stress_ratio: np.ndarray,
) -> dict[str, np.ndarray]:
    """Predict fracture orientation for tensile failure."""
    # Tensile fractures form perpendicular to minimum stress
    fracture_strike = (shmin_azimuth + 90.0) % 360.0
    fracture_dip = np.full_like(fracture_strike, 90.0)
    fracture_type = np.full(len(fracture_strike), "tensile", dtype=object)

    return {
        "strike": fracture_strike,
        "dip": fracture_dip,
        "type": fracture_type,
        "azimuth": fracture_strike,
    }


def fracture_orientation_distribution(
    mean_strike: float,
    concentration: float = 10.0,
    n_samples: int = 1000,
) -> np.ndarray:
    """Generate fracture orientation distribution using von Mises distribution.

    Useful for modeling natural fracture networks with preferred orientation.

    Args:
        mean_strike: Mean fracture strike in degrees.
        concentration: Concentration parameter (higher = more clustered).
        n_samples: Number of samples to generate.

    Returns:
        Array of fracture strikes in degrees.

    Example:
        >>> from geosmith.primitives.geomechanics import fracture_orientation_distribution
        >>>
        >>> strikes = fracture_orientation_distribution(mean_strike=45.0, concentration=10.0, n_samples=1000)
        >>> print(f"Generated {len(strikes)} fracture orientations")
    """
    # Optional scipy dependency
    try:
        from scipy.stats import vonmises

        SCIPY_AVAILABLE = True
    except ImportError:
        SCIPY_AVAILABLE = False

    if not SCIPY_AVAILABLE:
        raise ImportError(
            "scipy is required for fracture orientation distribution. "
            "Install with: pip install geosmith[primitives] or pip install scipy"
        )

    kappa = concentration
    strikes_rad = vonmises.rvs(kappa, loc=np.radians(mean_strike), size=n_samples)
    strikes_deg = np.degrees(strikes_rad) % 360.0

    return strikes_deg


def calculate_fracture_aperture(
    normal_stress: Union[np.ndarray, float],
    closure_stress: float = 5.0,
    initial_aperture: float = 0.1,
    stiffness: float = 10.0,
) -> Union[np.ndarray, float]:
    """Calculate fracture aperture under normal stress.

    Uses linear elastic model: aperture decreases with increasing normal stress.

    Args:
        normal_stress: Normal stress acting on fracture (MPa).
        closure_stress: Stress at which fracture closes (MPa), default 5.0.
        initial_aperture: Initial aperture at zero stress (mm), default 0.1.
        stiffness: Fracture stiffness (MPa/mm), default 10.0.

    Returns:
        Fracture aperture in mm.

    Example:
        >>> from geosmith.primitives.geomechanics import calculate_fracture_aperture
        >>>
        >>> aperture = calculate_fracture_aperture(normal_stress=20.0, closure_stress=5.0)
        >>> print(f"Fracture aperture: {aperture:.3f} mm")
    """
    normal_stress = np.asarray(normal_stress, dtype=float)

    # Linear closure model
    aperture = initial_aperture - (normal_stress - closure_stress) / stiffness
    aperture = np.clip(aperture, 0.0, initial_aperture)

    # Return scalar if input was scalar
    if normal_stress.ndim == 0:
        return float(aperture)
    return aperture


def calculate_fracture_permeability(
    aperture: Union[np.ndarray, float],
    spacing: float = 1.0,
    viscosity: float = 1.0e-3,
) -> Union[np.ndarray, float]:
    """Calculate fracture permeability using cubic law.

    k = (aperture^3) / (12 * spacing)

    Args:
        aperture: Fracture aperture (mm).
        spacing: Fracture spacing (m), default 1.0.
        viscosity: Fluid viscosity (Pa·s), default 1.0e-3.

    Returns:
        Permeability in mD.

    Example:
        >>> from geosmith.primitives.geomechanics import calculate_fracture_permeability
        >>>
        >>> k = calculate_fracture_permeability(aperture=0.1, spacing=1.0)
        >>> print(f"Fracture permeability: {k:.2f} mD")
    """
    aperture = np.asarray(aperture, dtype=float)
    aperture_m = aperture * 1e-3  # Convert mm to m

    # Cubic law: k = b^3 / (12 * s)
    k_m2 = (aperture_m**3) / (12 * spacing)

    # Convert to mD
    k_md = k_m2 * 1.01325e15

    # Return scalar if input was scalar
    if aperture.ndim == 0:
        return float(k_md)
    return k_md



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

