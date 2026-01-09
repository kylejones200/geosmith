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

