"""Geomechanics calculation primitives.

Pure geomechanics operations.
Migrated from geosuite.geomech.stresses.
Layer 2: Primitives - Pure operations.
"""

from typing import Union

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

