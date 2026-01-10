"""Geomechanics: Pressure calculations

Pure geomechanics operations - pressure module.
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

    This function is JIT-compiled for 2-5x speedup.

    Args:
        pressure: Pressure array (MPa).
        depth: Depth array (meters).

    Returns:
        Pressure gradient (MPa/m).
    """
    from geosmith.primitives._numba_helpers import njit

    @njit(cache=True)
    def _kernel(p: np.ndarray, d: np.ndarray) -> np.ndarray:
        n = len(p)
        gradient = np.zeros(n, dtype=np.float64)

        for i in range(1, n):
            dz = d[i] - d[i - 1]
            if dz > 0.0:
                gradient[i] = (p[i] - p[i - 1]) / dz
            else:
                gradient[i] = gradient[i - 1] if i > 1 else 0.0

        # Extrapolate first value
        gradient[0] = gradient[1] if n > 1 else 0.0

        return gradient

    return _kernel(pressure, depth)

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

def _calculate_overburden_stress_kernel(
    depth: np.ndarray, rhob_kg: np.ndarray, g: float
) -> np.ndarray:
    """Numba-optimized kernel for overburden stress integration.

    This function is JIT-compiled for 20-50x speedup on large datasets.
    Uses trapezoidal integration: Sv = integral(rho * g * dz)

    Args:
        depth: Depth array (meters).
        rhob_kg: Bulk density array (kg/m³).
        g: Gravitational acceleration (m/s²).

    Returns:
        Overburden stress (MPa).
    """
    from geosmith.primitives._numba_helpers import njit

    @njit(cache=True)
    def _kernel(d: np.ndarray, r: np.ndarray, grav: float) -> np.ndarray:
        n = len(d)
        sv = np.zeros(n, dtype=np.float64)

        # Trapezoidal integration
        for i in range(1, n):
            dz = d[i] - d[i - 1]
            if dz > 0.0:
                avg_rho = (r[i] + r[i - 1]) * 0.5
                sv[i] = sv[i - 1] + avg_rho * grav * dz * 1e-6  # Convert Pa to MPa
            else:
                sv[i] = sv[i - 1] if i > 1 else 0.0

        return sv

    return _kernel(depth, rhob_kg, g)

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

def pore_pressure_eaton(
    sv: Union[np.ndarray, float, "pd.Series"],
    porosity: Union[np.ndarray, float, "pd.Series"],
    phi_initial: float = 0.35,
    beta: float = 0.03,
) -> Union[np.ndarray, "pd.Series"]:
    """Compute pore pressure using Eaton's method based on porosity.

    Pp = Sv - (ln(φ / φ_initial) / -beta)

    Where φ is porosity and φ_initial is initial porosity at surface.

    Args:
        sv: Vertical stress (MPa).
        porosity: Porosity as fraction (0-1).
        phi_initial: Initial porosity at surface, default 0.35.
        beta: Compaction coefficient (1/MPa), default 0.03.

    Returns:
        Pore pressure in MPa.

    Example:
        >>> from geosmith.primitives.geomechanics import pore_pressure_eaton
        >>> import numpy as np
        >>>
        >>> sv = np.array([50.0, 60.0, 70.0])
        >>> porosity = np.array([0.25, 0.20, 0.15])
        >>> pp = pore_pressure_eaton(sv, porosity, phi_initial=0.35, beta=0.03)
        >>> print(f"Pore pressure range: {pp.min():.1f} - {pp.max():.1f} MPa")
    """
    if not PANDAS_AVAILABLE:
        raise ImportError(
            "pandas is required for pore_pressure_eaton. "
            "Install with: pip install pandas"
        )

    sv_orig = sv
    porosity_orig = porosity

    sv = np.asarray(sv, dtype=np.float64)
    porosity = np.asarray(porosity, dtype=np.float64)

    if len(sv) != len(porosity):
        raise ValueError("sv and porosity must have same length")

    # Clip porosity to avoid log(0) issues
    porosity = np.clip(porosity, 1e-6, 0.99)

    phi_ratio = porosity / phi_initial
    phi_ratio = np.clip(phi_ratio, 1e-6, None)  # Avoid log of negative or zero

    pp = sv - (np.log(phi_ratio) / (-beta))

    # Return same type as input
    if isinstance(sv_orig, pd.Series):
        return pd.Series(pp, index=sv_orig.index)
    elif isinstance(porosity_orig, pd.Series):
        return pd.Series(pp, index=porosity_orig.index)
    elif np.ndim(sv) == 0:
        return float(pp)
    return pp

def sv_from_density(
    depth: Union[np.ndarray, float, "pd.Series"],
    rhob: Union[np.ndarray, float, "pd.Series"],
    g: Optional[float] = None,
) -> Union[np.ndarray, "pd.Series"]:
    """Compute vertical stress Sv by integrating density over depth.

    Uses trapezoidal integration to compute overburden stress from density log.

    Args:
        depth: Depth array in meters (monotonic increasing).
        rhob: Bulk density in g/cc.
        g: Gravitational acceleration (m/s²), default 9.80665.

    Returns:
        Sv in MPa (overburden stress).

    Example:
        >>> from geosmith.primitives.geomechanics import sv_from_density
        >>> import numpy as np
        >>>
        >>> depth = np.array([0, 1000, 2000, 3000])
        >>> rhob = np.array([2.0, 2.2, 2.4, 2.6])
        >>> sv = sv_from_density(depth, rhob)
        >>> print(f"Overburden at {depth[-1]}m: {sv[-1]:.1f} MPa")
    """
    if not PANDAS_AVAILABLE:
        raise ImportError(
            "pandas is required for sv_from_density. "
            "Install with: pip install pandas"
        )

    depth_orig = depth
    rhob_orig = rhob

    if g is None:
        g = 9.80665  # m/s^2

    depth = np.asarray(depth, dtype=np.float64)
    rhob = np.asarray(rhob, dtype=np.float64)

    if len(depth) < 2:
        if isinstance(depth_orig, pd.Series):
            return pd.Series(np.zeros_like(depth, dtype=float), index=depth_orig.index)
        return np.zeros_like(depth, dtype=float)

    if len(depth) != len(rhob):
        raise ValueError("Depth and density arrays must have same length")

    # Convert g/cc to kg/m³
    rho_kg = rhob * 1000.0

    # Trapezoidal integration
    dz = np.diff(depth)
    rho_mid = 0.5 * (rho_kg[1:] + rho_kg[:-1])

    # Incremental stress in Pa
    dsv = rho_mid * g * dz

    # Cumulative stress in Pa
    sv_pa = np.concatenate([[0.0], np.cumsum(dsv)])

    # Convert to MPa
    sv_mpa = sv_pa / 1e6

    # Return same type as input
    if isinstance(depth_orig, pd.Series):
        return pd.Series(sv_mpa, index=depth_orig.index)
    elif isinstance(rhob_orig, pd.Series):
        return pd.Series(sv_mpa, index=rhob_orig.index)
    return sv_mpa

def mud_weight_equivalent(
    pressure: Union[np.ndarray, float, "pd.Series"],
    depth: Union[np.ndarray, float, "pd.Series"],
) -> Union[np.ndarray, "pd.Series"]:
    """Convert pressure to equivalent mud weight (EMW).

    EMW (g/cc) = Pressure (MPa) / (Depth (m) * 0.00981)

    Args:
        pressure: Pressure in MPa.
        depth: Depth in meters.

    Returns:
        Equivalent mud weight in g/cc.

    Example:
        >>> from geosmith.primitives.geomechanics import mud_weight_equivalent
        >>> import numpy as np
        >>>
        >>> pressure = np.array([20.0, 30.0, 40.0])
        >>> depth = np.array([2000.0, 3000.0, 4000.0])
        >>> mw = mud_weight_equivalent(pressure, depth)
        >>> print(f"Mud weight range: {mw.min():.2f} - {mw.max():.2f} g/cc")
    """
    if not PANDAS_AVAILABLE:
        raise ImportError(
            "pandas is required for mud_weight_equivalent. "
            "Install with: pip install pandas"
        )

    pressure_orig = pressure
    depth_orig = depth

    pressure = np.asarray(pressure, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)

    if len(pressure) != len(depth):
        raise ValueError("Pressure and depth arrays must have same length")

    # Avoid division by zero
    depth_nonzero = np.where(depth == 0, 1e-6, depth)
    emw = pressure / (depth_nonzero * 0.00981)

    # Clip to reasonable range (0.8 - 2.5 g/cc)
    emw = np.clip(emw, 0.8, 2.5)

    # Return same type as input
    if isinstance(pressure_orig, pd.Series):
        return pd.Series(emw, index=pressure_orig.index)
    elif isinstance(depth_orig, pd.Series):
        return pd.Series(emw, index=depth_orig.index)
    elif np.ndim(pressure) == 0:
        return float(emw)
    return emw

def _calculate_overburden_stress_kernel(
    depth: np.ndarray,
    rhob_kg: np.ndarray,
    g: float,
) -> np.ndarray:
    """Numba-optimized kernel for overburden stress integration.

    This function is JIT-compiled for 20-50x speedup on large datasets.
    Uses trapezoidal integration: Sv = integral(rho * g * dz)

    Args:
        depth: Depth array (meters).
        rhob_kg: Bulk density array (kg/m³).
        g: Gravitational acceleration (m/s²).

    Returns:
        Overburden stress (MPa).
    """
    from geosmith.primitives._numba_helpers import njit

    @njit(cache=True)
    def _kernel(d: np.ndarray, r: np.ndarray, grav: float) -> np.ndarray:
        n = len(d)
        sv = np.zeros(n, dtype=np.float64)

        # Trapezoidal integration
        for i in range(1, n):
            dz = d[i] - d[i - 1]
            if dz > 0.0:
                avg_rho = (r[i] + r[i - 1]) * 0.5
                sv[i] = sv[i - 1] + avg_rho * grav * dz * 1e-6  # Convert Pa to MPa
            else:
                sv[i] = sv[i - 1] if i > 1 else 0.0

        return sv

    return _kernel(depth, rhob_kg, g)

def calculate_pore_pressure_eaton_sonic(
    depth: Union[np.ndarray, float, "pd.Series"],
    dt: Union[np.ndarray, float, "pd.Series"],
    dt_normal: Union[np.ndarray, float, "pd.Series"],
    sv: Union[np.ndarray, float, "pd.Series"],
    ph: Union[np.ndarray, float, "pd.Series"],
    exponent: float = 3.0,
) -> Union[np.ndarray, "pd.Series"]:
    """Calculate pore pressure using Eaton's method with sonic data.

    Pp = Sv - (Sv - Ph) * (dt_normal / dt)^exponent

    This is Eaton's method specifically for sonic transit time data.

    Args:
        depth: Depth array (meters).
        dt: Measured sonic transit time (us/ft).
        dt_normal: Normal compaction trend sonic (us/ft).
        sv: Overburden stress (MPa).
        ph: Hydrostatic pressure (MPa).
        exponent: Eaton exponent, default 3.0 (typically 3.0 for sonic).

    Returns:
        Pore pressure (MPa).

    Example:
        >>> from geosmith.primitives.geomechanics import calculate_pore_pressure_eaton_sonic
        >>> import numpy as np
        >>>
        >>> depth = np.array([1000, 2000, 3000])
        >>> dt = np.array([80, 85, 90])  # us/ft
        >>> dt_normal = np.array([70, 72, 74])
        >>> sv = np.array([25, 50, 75])
        >>> ph = np.array([10, 20, 30])
        >>> pp = calculate_pore_pressure_eaton_sonic(depth, dt, dt_normal, sv, ph)
        >>> print(f"Pore pressure range: {pp.min():.1f} - {pp.max():.1f} MPa")
    """
    if not PANDAS_AVAILABLE:
        raise ImportError(
            "pandas is required for calculate_pore_pressure_eaton_sonic. "
            "Install with: pip install pandas"
        )

    depth_orig = depth
    dt_orig = dt
    dt_normal_orig = dt_normal
    sv_orig = sv
    ph_orig = ph

    depth = np.asarray(depth, dtype=np.float64)
    dt = np.asarray(dt, dtype=np.float64)
    dt_normal = np.asarray(dt_normal, dtype=np.float64)
    sv = np.asarray(sv, dtype=np.float64)
    ph = np.asarray(ph, dtype=np.float64)

    arrays = [depth, dt, dt_normal, sv, ph]
    if any(len(arr) == 0 for arr in arrays):
        raise ValueError("All input arrays must not be empty")

    lengths = [len(arr) for arr in arrays]
    if len(set(lengths)) > 1:
        raise ValueError(
            f"All input arrays must have the same length. Got lengths: {lengths}"
        )

    # Avoid division by zero
    dt = np.where(dt <= 0, np.nan, dt)
    dt_normal = np.where(dt_normal <= 0, np.nan, dt_normal)

    pp = sv - (sv - ph) * (dt_normal / dt) ** exponent

    # Return same type as input
    if isinstance(depth_orig, pd.Series):
        return pd.Series(pp, index=depth_orig.index)
    elif isinstance(dt_orig, pd.Series):
        return pd.Series(pp, index=dt_orig.index)
    elif np.ndim(sv) == 0:
        return float(pp)
    return pp

def calculate_pore_pressure_bowers(
    depth: Union[np.ndarray, float, "pd.Series"],
    dt: Union[np.ndarray, float, "pd.Series"],
    dt_ml: float = 100.0,
    A: float = 5.0,
    B: float = 1.2,
    sv: Optional[Union[np.ndarray, float, "pd.Series"]] = None,
    ph: Optional[Union[np.ndarray, float, "pd.Series"]] = None,
    rho_water: float = 1.03,
    g: float = 9.80665,
) -> Union[np.ndarray, "pd.Series"]:
    """Calculate pore pressure using Bowers' method.

    This method accounts for unloading due to uplift/erosion and is more
    sophisticated than Eaton's method for complex geological histories.

    Args:
        depth: Depth array (meters).
        dt: Measured sonic transit time (us/ft).
        dt_ml: Mudline sonic (us/ft), default 100.0.
        A: Bowers A parameter, default 5.0.
        B: Bowers B parameter, default 1.2.
        sv: Overburden stress (MPa), calculated if not provided.
        ph: Hydrostatic pressure (MPa), calculated if not provided.
        rho_water: Water density (g/cc), default 1.03.
        g: Gravitational acceleration (m/s²), default 9.80665.

    Returns:
        Pore pressure (MPa).

    Example:
        >>> from geosmith.primitives.geomechanics import calculate_pore_pressure_bowers
        >>> import numpy as np
        >>>
        >>> depth = np.array([1000, 2000, 3000])
        >>> dt = np.array([80, 85, 90])
        >>> pp = calculate_pore_pressure_bowers(depth, dt, dt_ml=100.0)
        >>> print(f"Pore pressure range: {pp.min():.1f} - {pp.max():.1f} MPa")
    """
    if not PANDAS_AVAILABLE:
        raise ImportError(
            "pandas is required for calculate_pore_pressure_bowers. "
            "Install with: pip install pandas"
        )

    depth_orig = depth
    dt_orig = dt

    depth = np.asarray(depth, dtype=np.float64)
    dt = np.asarray(dt, dtype=np.float64)

    if len(depth) == 0 or len(dt) == 0:
        raise ValueError("Depth and sonic transit time arrays must not be empty")

    if len(depth) != len(dt):
        raise ValueError("Depth and sonic transit time arrays must have same length")

    if sv is None:
        # Assume average overburden gradient (0.023 MPa/m)
        sv = 0.023 * depth  # MPa
    else:
        sv = np.asarray(sv, dtype=np.float64)
        if len(sv) != len(depth):
            raise ValueError("Overburden stress array must have same length as depth")

    if ph is None:
        ph = calculate_hydrostatic_pressure(depth, rho_water=rho_water, g=g)
    else:
        ph = np.asarray(ph, dtype=np.float64)
        if len(ph) != len(depth):
            raise ValueError(
                "Hydrostatic pressure array must have same length as depth"
            )

    # Calculate effective stress from sonic using Bowers' relationship
    sigma_eff = ((dt - dt_ml) / A) ** (1 / B)

    # Pore pressure
    pp = sv - sigma_eff

    # Return same type as input
    if isinstance(depth_orig, pd.Series):
        return pd.Series(pp, index=depth_orig.index)
    elif isinstance(dt_orig, pd.Series):
        return pd.Series(pp, index=dt_orig.index)
    elif np.ndim(depth) == 0:
        return float(pp)
    return pp

def _calculate_pressure_gradient_kernel(
    pressure: np.ndarray, depth: np.ndarray
) -> np.ndarray:
    """Numba-optimized kernel for pressure gradient calculation.

    This function is JIT-compiled for 2-5x speedup.

    Args:
        pressure: Pressure array (MPa).
        depth: Depth array (meters).

    Returns:
        Pressure gradient (MPa/m).
    """
    from geosmith.primitives._numba_helpers import njit

    @njit(cache=True)
    def _kernel(p: np.ndarray, d: np.ndarray) -> np.ndarray:
        n = len(p)
        gradient = np.zeros(n, dtype=np.float64)

        for i in range(1, n):
            dz = d[i] - d[i - 1]
            if dz > 0.0:
                gradient[i] = (p[i] - p[i - 1]) / dz
            else:
                gradient[i] = gradient[i - 1] if i > 1 else 0.0

        # Extrapolate first value
        gradient[0] = gradient[1] if n > 1 else 0.0

        return gradient

    return _kernel(pressure, depth)

