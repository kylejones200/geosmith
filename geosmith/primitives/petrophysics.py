"""Petrophysics calculation primitives.

Pure petrophysics operations.
Migrated from geosuite.petro.archie.
Layer 2: Primitives - Pure operations.
"""

from dataclasses import dataclass
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


@dataclass(frozen=True)
class ArchieParams:
    """Archie equation parameters.

    Attributes:
        a: Tortuosity factor (typically 0.6-1.0).
        m: Cementation exponent (typically 1.8-2.5).
        n: Saturation exponent (typically 2.0).
        rw: Water resistivity (ohm·m).
    """

    a: float = 1.0
    m: float = 2.0
    n: float = 2.0
    rw: float = 0.1

    def __post_init__(self) -> None:
        """Validate Archie parameters."""
        if self.a <= 0:
            raise ValueError(f"Tortuosity factor a must be > 0, got {self.a}")

        if self.m <= 0:
            raise ValueError(f"Cementation exponent m must be > 0, got {self.m}")

        if self.n <= 0:
            raise ValueError(f"Saturation exponent n must be > 0, got {self.n}")

        if self.rw <= 0:
            raise ValueError(f"Water resistivity rw must be > 0, got {self.rw}")


def calculate_water_saturation(
    rt: Union[np.ndarray, float],
    phi: Union[np.ndarray, float],
    params: ArchieParams,
) -> np.ndarray:
    """Calculate water saturation using Archie's equation.

    Sw^n = (a * Rw) / (Rt * phi^m)  => Sw = [(a*Rw)/(Rt*phi^m)]^(1/n)

    Args:
        rt: True formation resistivity (ohm·m).
        phi: Porosity (fraction, 0-1).
        params: Archie parameters.

    Returns:
        Water saturation (fraction, 0-1).

    Example:
        >>> from geosmith.primitives.petrophysics import calculate_water_saturation, ArchieParams
        >>>
        >>> params = ArchieParams(a=1.0, m=2.0, n=2.0, rw=0.1)
        >>> sw = calculate_water_saturation(rt=10.0, phi=0.25, params=params)
        >>> print(f"Water saturation: {sw:.2%}")
    """
    rt = np.asarray(rt, dtype=float)
    phi = np.asarray(phi, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        denom = rt * np.power(phi, params.m)
        x = (params.a * params.rw) / np.where(denom == 0, np.nan, denom)
        sw = np.power(np.clip(x, 1e-12, 1e6), 1.0 / params.n)

    return np.nan_to_num(sw, nan=np.nan, posinf=np.nan, neginf=np.nan)


def calculate_bulk_volume_water(
    phi: Union[np.ndarray, float],
    sw: Union[np.ndarray, float],
) -> np.ndarray:
    """Calculate bulk volume water (BVW).

    BVW = phi * Sw

    Args:
        phi: Porosity (fraction, 0-1).
        sw: Water saturation (fraction, 0-1).

    Returns:
        Bulk volume water (fraction, 0-1).

    Example:
        >>> from geosmith.primitives.petrophysics import calculate_bulk_volume_water
        >>>
        >>> bvw = calculate_bulk_volume_water(phi=0.25, sw=0.5)
        >>> print(f"Bulk volume water: {bvw:.2%}")
    """
    return np.asarray(phi, dtype=float) * np.asarray(sw, dtype=float)


@njit(cache=True)
def _pickett_isolines_kernel(
    phi_grid: np.ndarray,
    sw_vals: np.ndarray,
    a: float,
    m: float,
    n: float,
    rw: float,
    rt_min: float,
    rt_max: float,
) -> np.ndarray:
    """Numba-optimized kernel for computing Pickett plot isolines.

    Args:
        phi_grid: Porosity values for isoline.
        sw_vals: Water saturation values for each isoline.
        a, m, n, rw: Archie parameters.
        rt_min, rt_max: Resistivity bounds.

    Returns:
        2D array of resistivity values (n_isolines x n_points).
    """
    n_isolines = len(sw_vals)
    n_points = len(phi_grid)
    rt_array = np.zeros((n_isolines, n_points), dtype=np.float64)

    for i in range(n_isolines):
        sw = sw_vals[i]
        for j in range(n_points):
            phi = phi_grid[j]
            rt = (a * rw) / (phi**m * sw**n)
            # Clip to bounds
            if rt < rt_min:
                rt = rt_min
            elif rt > rt_max:
                rt = rt_max
            rt_array[i, j] = rt

    return rt_array


def pickett_isolines(
    phi_vals: np.ndarray,
    sw_vals: np.ndarray,
    params: ArchieParams,
    rt_min: float = 0.1,
    rt_max: float = 1000.0,
    num_points: int = 100,
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Generate isolines for a Pickett plot (log-log Rt vs Phi) at constant Sw.

    Args:
        phi_vals: Porosity values to span (used to determine grid range).
        sw_vals: Water saturation values for each isoline.
        params: Archie parameters.
        rt_min: Minimum resistivity to clip.
        rt_max: Maximum resistivity to clip.
        num_points: Number of points per isoline.

    Returns:
        List of (phi_array, rt_array, label) tuples for each Sw in sw_vals.

    Example:
        >>> from geosmith.primitives.petrophysics import pickett_isolines, ArchieParams
        >>>
        >>> params = ArchieParams()
        >>> isolines = pickett_isolines(
        ...     phi_vals=np.array([0.1, 0.3]),
        ...     sw_vals=np.array([0.5, 0.7, 1.0]),
        ...     params=params
        ... )
        >>> for phi, rt, label in isolines:
        ...     print(f"{label}: {len(phi)} points")
    """
    # Generate porosity grid
    phi_min = max(1e-4, np.min(phi_vals))
    phi_max = min(0.5, np.max(phi_vals))
    phi_grid = np.logspace(np.log10(phi_min), np.log10(phi_max), num_points)

    # Convert sw_vals to numpy array
    sw_array = np.asarray(sw_vals, dtype=np.float64)

    # Call optimized kernel
    rt_array = _pickett_isolines_kernel(
        phi_grid,
        sw_array,
        params.a,
        params.m,
        params.n,
        params.rw,
        rt_min,
        rt_max,
    )

    # Build output list
    lines: list[tuple[np.ndarray, np.ndarray, str]] = []
    for i, sw in enumerate(sw_array):
        lines.append((phi_grid, rt_array[i], f"Sw={sw:g}"))

    return lines

