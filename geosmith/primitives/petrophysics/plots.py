"""Geosmith petrophysics: Plotting utilities (Pickett plot isolines)

Migrated from geosuite.petro.
Layer 2: Primitives - Pure operations.
"""

from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from geosmith.primitives.petrophysics._common import logger, njit

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
