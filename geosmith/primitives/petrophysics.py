"""Petrophysics calculation primitives.

Pure petrophysics operations.
Migrated from geosuite.petro.archie, geosuite.petro.permeability,
geosuite.petro.shaly_sand, and geosuite.petro.rock_physics.
Layer 2: Primitives - Pure operations.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

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


def calculate_permeability_kozeny_carman(
    phi: Union[np.ndarray, float],
    sw: Optional[Union[np.ndarray, float]] = None,
    grain_size_microns: float = 100.0,
    shape_factor: float = 2.5,
    tortuosity: float = 1.0,
) -> np.ndarray:
    """Calculate permeability using Kozeny-Carman equation.

    k = (phi^3 * d^2) / (180 * (1 - phi)^2 * tau^2 * F)

    Args:
        phi: Porosity (fraction, 0-1).
        sw: Optional water saturation (fraction, 0-1).
        grain_size_microns: Average grain size in microns (default 100).
        shape_factor: Shape factor (default 2.5).
        tortuosity: Tortuosity factor (default 1.0).

    Returns:
        Permeability in millidarcies (mD).

    Example:
        >>> from geosmith.primitives.petrophysics import calculate_permeability_kozeny_carman
        >>>
        >>> k = calculate_permeability_kozeny_carman(phi=0.25, sw=0.5)
        >>> print(f"Permeability: {k:.2f} mD")
    """
    phi = np.asarray(phi, dtype=float)

    if phi.size == 0:
        raise ValueError("Porosity array must not be empty")

    if np.any((phi < 0) | (phi > 1)):
        logger.warning("Porosity values outside [0, 1] range detected")

    phi = np.clip(phi, 0.01, 0.99)

    d_meters = grain_size_microns * 1e-6
    d_cm = d_meters * 100

    k_cm2 = (
        phi**3
        * d_cm**2
        / (180 * (1 - phi) ** 2 * tortuosity**2 * shape_factor)
    )

    k_md = k_cm2 * 1.01325e8

    if sw is not None:
        sw = np.asarray(sw, dtype=float)
        if sw.size != phi.size:
            raise ValueError(
                "Water saturation and porosity arrays must have same length"
            )
        k_md = k_md * (1 - sw) ** 2

    k_md = np.clip(k_md, 0.001, 1e6)

    return k_md


def calculate_permeability_timur(
    phi: Union[np.ndarray, float],
    sw: Union[np.ndarray, float],
    coefficient: float = 0.136,
    porosity_exponent: float = 4.4,
    saturation_exponent: float = 2.0,
) -> np.ndarray:
    """Calculate permeability using Timur equation.

    k = C * (phi^a) / (sw^b)

    Default coefficients from Timur (1968) for sandstones.

    Args:
        phi: Porosity (fraction, 0-1).
        sw: Water saturation (fraction, 0-1).
        coefficient: Coefficient C (default 0.136).
        porosity_exponent: Exponent a (default 4.4).
        saturation_exponent: Exponent b (default 2.0).

    Returns:
        Permeability in millidarcies (mD).

    Example:
        >>> from geosmith.primitives.petrophysics import calculate_permeability_timur
        >>>
        >>> k = calculate_permeability_timur(phi=0.25, sw=0.5)
        >>> print(f"Permeability: {k:.2f} mD")
    """
    phi = np.asarray(phi, dtype=float)
    sw = np.asarray(sw, dtype=float)

    if phi.size == 0 or sw.size == 0:
        raise ValueError(
            "Porosity and water saturation arrays must not be empty"
        )

    if phi.size != sw.size:
        raise ValueError(
            "Porosity and water saturation arrays must have same length"
        )

    phi = np.clip(phi, 0.01, 0.99)
    sw = np.clip(sw, 0.01, 0.99)

    k_md = coefficient * (phi**porosity_exponent) / (sw**saturation_exponent)

    k_md = np.clip(k_md, 0.001, 1e6)

    return k_md


def calculate_permeability_porosity_only(
    phi: Union[np.ndarray, float],
    coefficient: float = 100.0,
    exponent: float = 3.0,
) -> np.ndarray:
    """Calculate permeability from porosity only (simple power law).

    k = C * phi^a

    Useful when water saturation is not available.

    Args:
        phi: Porosity (fraction, 0-1).
        coefficient: Coefficient C (default 100.0).
        exponent: Exponent a (default 3.0).

    Returns:
        Permeability in millidarcies (mD).

    Example:
        >>> from geosmith.primitives.petrophysics import calculate_permeability_porosity_only
        >>>
        >>> k = calculate_permeability_porosity_only(phi=0.25)
        >>> print(f"Permeability: {k:.2f} mD")
    """
    phi = np.asarray(phi, dtype=float)

    if phi.size == 0:
        raise ValueError("Porosity array must not be empty")

    phi = np.clip(phi, 0.01, 0.99)

    k_md = coefficient * (phi**exponent)

    k_md = np.clip(k_md, 0.001, 1e6)

    return k_md


# Shaly Sand Water Saturation Models
# Migrated from geosuite.petro.shaly_sand


def calculate_water_saturation_simandoux(
    phi: Union[np.ndarray, float],
    rt: Union[np.ndarray, float],
    rsh: Union[np.ndarray, float],
    vsh: Union[np.ndarray, float],
    rw: float = 0.05,
    m: float = 2.0,
    n: float = 2.0,
    a: float = 1.0,
) -> np.ndarray:
    """Calculate water saturation using Simandoux equation for shaly sands.

    The Simandoux equation accounts for clay conductivity in shaly formations:
    Sw = sqrt((a * Rw) / (phi^m * (1/Rt - Vsh/Rsh)))

    Args:
        phi: Porosity (fraction).
        rt: True resistivity (ohm-m).
        rsh: Shale resistivity (ohm-m).
        vsh: Shale volume fraction (fraction).
        rw: Formation water resistivity (ohm-m, default 0.05).
        m: Cementation exponent (default 2.0).
        n: Saturation exponent (default 2.0, typically 2.0 for Simandoux).
        a: Tortuosity factor (default 1.0).

    Returns:
        Water saturation (fraction).

    Example:
        >>> from geosmith.primitives.petrophysics import calculate_water_saturation_simandoux
        >>>
        >>> sw = calculate_water_saturation_simandoux(
        ...     phi=0.25, rt=10.0, rsh=2.0, vsh=0.3, rw=0.05
        ... )
        >>> print(f"Water saturation: {sw:.2%}")
    """
    # Convert to numpy arrays
    phi = np.asarray(phi, dtype=float)
    rt = np.asarray(rt, dtype=float)
    rsh = np.asarray(rsh, dtype=float)
    vsh = np.asarray(vsh, dtype=float)

    # Validate inputs
    lengths = [len(phi), len(rt), len(rsh), len(vsh)]
    if len(set(lengths)) > 1:
        raise ValueError(
            f"All input arrays must have the same length. "
            f"Got lengths: phi={lengths[0]}, rt={lengths[1]}, rsh={lengths[2]}, vsh={lengths[3]}"
        )

    if np.any(phi <= 0):
        logger.warning("Found non-positive porosity values, will result in NaN")
    if np.any(rt <= 0):
        logger.warning("Found non-positive resistivity values, will result in NaN")
    if np.any(rsh <= 0):
        logger.warning("Found non-positive shale resistivity values, will result in NaN")
    if np.any((vsh < 0) | (vsh > 1)):
        logger.warning("Found shale volume outside [0, 1], clipping to valid range")
        vsh = np.clip(vsh, 0, 1)

    logger.debug(
        f"Calculating water saturation using Simandoux equation for {len(phi)} samples"
    )

    # Simandoux equation: Sw = sqrt((a * Rw) / (phi^m * (1/Rt - Vsh/Rsh)))
    with np.errstate(divide="ignore", invalid="ignore"):
        # Calculate clay conductivity term
        clay_conductivity = vsh / rsh

        # Calculate sand conductivity
        sand_conductivity = 1.0 / rt

        # Net conductivity (sand minus clay)
        net_conductivity = sand_conductivity - clay_conductivity

        # Avoid division by zero
        net_conductivity = np.where(net_conductivity <= 0, np.nan, net_conductivity)

        # Calculate water saturation
        numerator = a * rw
        denominator = (phi**m) * net_conductivity
        sw = np.sqrt(numerator / denominator)

    # Clip to valid range
    sw = np.clip(sw, 0, 1)

    # Count NaN values
    nan_count = np.isnan(sw).sum()
    if nan_count > 0:
        logger.warning(f"Generated {nan_count} NaN values in Simandoux calculation")

    return sw


def calculate_water_saturation_indonesia(
    phi: Union[np.ndarray, float],
    rt: Union[np.ndarray, float],
    rsh: Union[np.ndarray, float],
    vsh: Union[np.ndarray, float],
    rw: float = 0.05,
    m: float = 2.0,
    n: float = 2.0,
    a: float = 1.0,
) -> np.ndarray:
    """Calculate water saturation using Indonesia equation for shaly sands.

    The Indonesia equation is an improved version of Simandoux that better
    handles high shale volumes:
    Sw = [sqrt((a * Rw) / (phi^m * Rt)) + sqrt(Vsh * Rw / Rsh))]^(-2/n)

    Args:
        phi: Porosity (fraction).
        rt: True resistivity (ohm-m).
        rsh: Shale resistivity (ohm-m).
        vsh: Shale volume fraction (fraction).
        rw: Formation water resistivity (ohm-m, default 0.05).
        m: Cementation exponent (default 2.0).
        n: Saturation exponent (default 2.0).
        a: Tortuosity factor (default 1.0).

    Returns:
        Water saturation (fraction).

    Example:
        >>> from geosmith.primitives.petrophysics import calculate_water_saturation_indonesia
        >>>
        >>> sw = calculate_water_saturation_indonesia(
        ...     phi=0.25, rt=10.0, rsh=2.0, vsh=0.3, rw=0.05
        ... )
        >>> print(f"Water saturation: {sw:.2%}")
    """
    # Convert to numpy arrays
    phi = np.asarray(phi, dtype=float)
    rt = np.asarray(rt, dtype=float)
    rsh = np.asarray(rsh, dtype=float)
    vsh = np.asarray(vsh, dtype=float)

    # Validate inputs
    lengths = [len(phi), len(rt), len(rsh), len(vsh)]
    if len(set(lengths)) > 1:
        raise ValueError(
            f"All input arrays must have the same length. "
            f"Got lengths: phi={lengths[0]}, rt={lengths[1]}, rsh={lengths[2]}, vsh={lengths[3]}"
        )

    # Vectorized validation checks
    validation_checks = {
        "non-positive porosity": np.any(phi <= 0),
        "non-positive resistivity": np.any(rt <= 0),
        "non-positive shale resistivity": np.any(rsh <= 0),
        "shale volume out of range": np.any((vsh < 0) | (vsh > 1)),
    }

    for check_name, check_result in validation_checks.items():
        if check_result:
            logger.warning(f"Found {check_name}, may result in invalid values")

    vsh = np.clip(vsh, 0, 1)

    logger.debug(
        f"Calculating water saturation using Indonesia equation for {len(phi)} samples"
    )

    # Indonesia equation: Sw = [sqrt((a * Rw) / (phi^m * Rt)) + sqrt(Vsh * Rw / Rsh))]^(-2/n)
    with np.errstate(divide="ignore", invalid="ignore"):
        # Archie term
        archie_term = np.sqrt((a * rw) / ((phi**m) * rt))

        # Shale term
        shale_term = np.sqrt(vsh * rw / rsh)

        # Combined term
        combined = archie_term + shale_term

        # Avoid division by zero
        combined = np.where(combined <= 0, np.nan, combined)

        # Calculate water saturation
        sw = np.power(combined, -2.0 / n)

    # Clip to valid range
    sw = np.clip(sw, 0, 1)

    # Count NaN values
    nan_count = np.isnan(sw).sum()
    if nan_count > 0:
        logger.warning(f"Generated {nan_count} NaN values in Indonesia calculation")

    return sw


def calculate_water_saturation_waxman_smits(
    phi: Union[np.ndarray, float],
    rt: Union[np.ndarray, float],
    cec: Union[np.ndarray, float],
    rw: float = 0.05,
    m: float = 2.0,
    n: float = 2.0,
    a: float = 1.0,
    b: Optional[float] = None,
    temperature: float = 25.0,
) -> np.ndarray:
    """Calculate water saturation using Waxman-Smits equation for shaly sands.

    The Waxman-Smits model accounts for clay cation exchange capacity (CEC)
    and is more physically-based than Simandoux:
    Sw = [sqrt((a * Rw) / (phi^m * Rt)) + B * Qv * Rw]^(-2/n)

    where Qv is the cation exchange capacity per unit pore volume.

    Args:
        phi: Porosity (fraction).
        rt: True resistivity (ohm-m).
        cec: Cation exchange capacity (meq/100g).
        rw: Formation water resistivity (ohm-m, default 0.05).
        m: Cementation exponent (default 2.0).
        n: Saturation exponent (default 2.0).
        a: Tortuosity factor (default 1.0).
        b: Equivalent counterion conductance (mho/m per meq/ml). If None, calculated from temperature.
        temperature: Formation temperature (°C, default 25.0) for B calculation if b is None.

    Returns:
        Water saturation (fraction).

    Example:
        >>> from geosmith.primitives.petrophysics import calculate_water_saturation_waxman_smits
        >>>
        >>> sw = calculate_water_saturation_waxman_smits(
        ...     phi=0.25, rt=10.0, cec=5.0, rw=0.05
        ... )
        >>> print(f"Water saturation: {sw:.2%}")
    """
    # Convert to numpy arrays
    phi = np.asarray(phi, dtype=float)
    rt = np.asarray(rt, dtype=float)
    cec = np.asarray(cec, dtype=float)

    # Validate inputs
    lengths = [len(phi), len(rt), len(cec)]
    if len(set(lengths)) > 1:
        raise ValueError(
            f"All input arrays must have the same length. "
            f"Got lengths: phi={lengths[0]}, rt={lengths[1]}, cec={lengths[2]}"
        )

    # Vectorized validation checks
    validation_checks = {
        "non-positive porosity": np.any(phi <= 0),
        "non-positive resistivity": np.any(rt <= 0),
        "negative CEC": np.any(cec < 0),
    }

    for check_name, check_result in validation_checks.items():
        if check_result:
            logger.warning(f"Found {check_name}, may result in invalid values")

    logger.debug(
        f"Calculating water saturation using Waxman-Smits equation for {len(phi)} samples"
    )

    # Calculate B (equivalent counterion conductance) if not provided
    if b is None:
        # B = 4.6 * (1 - 0.6 * exp(-0.77 / Rw)) at 25°C
        # Temperature correction: B(T) = B(25) * (1 + 0.02 * (T - 25))
        b_25 = 4.6 * (1 - 0.6 * np.exp(-0.77 / rw))
        b_val = b_25 * (1 + 0.02 * (temperature - 25.0))
        logger.debug(f"Calculated B = {b_val:.4f} mho/m per meq/ml at {temperature}°C")
    else:
        b_val = b

    # Convert CEC to Qv (meq/ml pore volume)
    # Qv = CEC * (1 - phi) * rho_grain / (phi * 100)
    # Simplified: assume rho_grain = 2.65 g/cc
    rho_grain = 2.65
    qv = cec * (1 - phi) * rho_grain / (phi * 100.0)

    # Waxman-Smits equation: Sw = [sqrt((a * Rw) / (phi^m * Rt)) + B * Qv * Rw]^(-2/n)
    with np.errstate(divide="ignore", invalid="ignore"):
        # Archie term
        archie_term = np.sqrt((a * rw) / ((phi**m) * rt))

        # Clay conductivity term
        clay_term = b_val * qv * rw

        # Combined term
        combined = archie_term + clay_term

        # Avoid division by zero
        combined = np.where(combined <= 0, np.nan, combined)

        # Calculate water saturation
        sw = np.power(combined, -2.0 / n)

    # Clip to valid range
    sw = np.clip(sw, 0, 1)

    # Count NaN values
    nan_count = np.isnan(sw).sum()
    if nan_count > 0:
        logger.warning(f"Generated {nan_count} NaN values in Waxman-Smits calculation")

    return sw


# Rock Physics Functions
# Migrated from geosuite.petro.rock_physics


def gassmann_fluid_substitution(
    k_sat_initial: Union[np.ndarray, float],
    k_dry: Union[np.ndarray, float],
    k_mineral: Union[float, np.ndarray],
    k_fluid_initial: Union[float, np.ndarray],
    k_fluid_final: Union[float, np.ndarray],
    phi: Union[np.ndarray, float],
) -> np.ndarray:
    """Perform Gassmann fluid substitution to predict bulk modulus after fluid change.

    Gassmann's equation predicts how bulk modulus changes when pore fluid
    is replaced, assuming constant pore pressure and no chemical interactions.

    K_sat_final = K_dry + (1 - K_dry/K_mineral)^2 / (phi/K_fluid_final + (1-phi)/K_mineral - K_dry/K_mineral^2)

    Args:
        k_sat_initial: Initial saturated bulk modulus (GPa).
        k_dry: Dry frame bulk modulus (GPa).
        k_mineral: Mineral bulk modulus (GPa). Typical values:
            - Quartz: 37 GPa
            - Calcite: 77 GPa
            - Dolomite: 95 GPa
        k_fluid_initial: Initial fluid bulk modulus (GPa). Typical values:
            - Water: 2.2 GPa
            - Oil: 0.5-2.0 GPa
            - Gas: 0.01-0.1 GPa
        k_fluid_final: Final fluid bulk modulus (GPa).
        phi: Porosity (fraction).

    Returns:
        Final saturated bulk modulus (GPa).

    Example:
        >>> from geosmith.primitives.petrophysics import gassmann_fluid_substitution
        >>>
        >>> k_final = gassmann_fluid_substitution(
        ...     k_sat_initial=20.0, k_dry=15.0, k_mineral=37.0,
        ...     k_fluid_initial=2.2, k_fluid_final=0.05, phi=0.25
        ... )
        >>> print(f"Final bulk modulus: {k_final:.2f} GPa")
    """
    # Convert to numpy arrays
    phi = np.asarray(phi, dtype=float)
    k_sat_initial = np.asarray(k_sat_initial, dtype=float)
    k_dry = np.asarray(k_dry, dtype=float)

    # Broadcast scalars to arrays
    k_mineral = (
        np.full_like(phi, k_mineral)
        if isinstance(k_mineral, (int, float))
        else np.asarray(k_mineral, dtype=float)
    )
    k_fluid_initial = (
        np.full_like(phi, k_fluid_initial)
        if isinstance(k_fluid_initial, (int, float))
        else np.asarray(k_fluid_initial, dtype=float)
    )
    k_fluid_final = (
        np.full_like(phi, k_fluid_final)
        if isinstance(k_fluid_final, (int, float))
        else np.asarray(k_fluid_final, dtype=float)
    )

    # Validate inputs
    lengths = [
        len(k_sat_initial),
        len(k_dry),
        len(k_mineral),
        len(k_fluid_initial),
        len(k_fluid_final),
        len(phi),
    ]
    if len(set(lengths)) > 1:
        raise ValueError(
            f"All input arrays must have the same length. Got lengths: {lengths}"
        )

    validation_checks = {
        "porosity outside (0, 1)": np.any((phi <= 0) | (phi >= 1)),
        "non-positive bulk moduli": np.any((k_dry <= 0) | (k_mineral <= 0)),
    }

    for check_name, check_result in validation_checks.items():
        if check_result:
            logger.warning(f"Found {check_name}, results may be invalid")

    logger.debug(
        f"Performing Gassmann fluid substitution for {len(phi)} samples"
    )

    # Gassmann's equation
    # K_sat = K_dry + (1 - K_dry/K_mineral)^2 / (phi/K_fluid + (1-phi)/K_mineral - K_dry/K_mineral^2)
    with np.errstate(divide="ignore", invalid="ignore"):
        # Calculate denominator
        term1 = phi / k_fluid_final
        term2 = (1 - phi) / k_mineral
        term3 = k_dry / (k_mineral**2)
        denominator = term1 + term2 - term3

        # Avoid division by zero
        denominator = np.where(denominator <= 0, np.nan, denominator)

        # Calculate numerator
        numerator = (1 - k_dry / k_mineral) ** 2

        # Calculate final saturated bulk modulus
        k_sat_final = k_dry + numerator / denominator

    # Count NaN values
    nan_count = np.isnan(k_sat_final).sum()
    if nan_count > 0:
        logger.warning(f"Generated {nan_count} NaN values in Gassmann calculation")

    return k_sat_final


def calculate_fluid_bulk_modulus(
    sw: Union[np.ndarray, float],
    so: Optional[Union[np.ndarray, float]] = None,
    sg: Optional[Union[np.ndarray, float]] = None,
    k_water: float = 2.2,
    k_oil: float = 1.0,
    k_gas: float = 0.05,
    temperature: float = 25.0,
    pressure: float = 20.0,
) -> np.ndarray:
    """Calculate effective fluid bulk modulus from saturations.

    Uses Reuss average (isostress) for fluid mixing:
    K_fluid = 1 / (Sw/K_water + So/K_oil + Sg/K_gas)

    Args:
        sw: Water saturation (fraction).
        so: Oil saturation (fraction). If None, calculated as 1 - Sw - Sg.
        sg: Gas saturation (fraction). If None, assumed to be 0.
        k_water: Water bulk modulus (GPa) at standard conditions (default 2.2).
        k_oil: Oil bulk modulus (GPa) at standard conditions (default 1.0).
        k_gas: Gas bulk modulus (GPa) at standard conditions (default 0.05).
        temperature: Temperature (°C) for pressure correction (default 25.0).
        pressure: Pressure (MPa) for pressure correction (default 20.0).

    Returns:
        Effective fluid bulk modulus (GPa).

    Example:
        >>> from geosmith.primitives.petrophysics import calculate_fluid_bulk_modulus
        >>>
        >>> k_fluid = calculate_fluid_bulk_modulus(sw=0.3, so=0.5, sg=0.2)
        >>> print(f"Fluid bulk modulus: {k_fluid:.2f} GPa")
    """
    sw = np.asarray(sw, dtype=float)

    # Calculate missing saturations
    sg = np.zeros_like(sw) if sg is None else np.asarray(sg, dtype=float)
    so = (1.0 - sw - sg) if so is None else np.asarray(so, dtype=float)

    # Validate and clip saturations
    saturation_bounds = np.any(
        [(sw < 0) | (sw > 1), (so < 0) | (so > 1), (sg < 0) | (sg > 1)]
    )
    if saturation_bounds:
        logger.warning("Found saturations outside [0, 1], clipping to valid range")

    sw, so, sg = np.clip(sw, 0, 1), np.clip(so, 0, 1), np.clip(sg, 0, 1)

    # Normalize to ensure Sw + So + Sg = 1
    total_sat = sw + so + sg
    sw = sw / total_sat
    so = so / total_sat
    sg = sg / total_sat

    # Apply pressure/temperature corrections (simplified)
    # Gas is most sensitive to pressure
    k_gas_corrected = k_gas * (1 + 0.01 * pressure)  # Rough correction

    # Reuss average (isostress)
    with np.errstate(divide="ignore", invalid="ignore"):
        k_fluid = 1.0 / (sw / k_water + so / k_oil + sg / k_gas_corrected)

    # Handle invalid values
    k_fluid = np.where(np.isfinite(k_fluid), k_fluid, np.nan)

    logger.debug(f"Calculated fluid bulk modulus for {len(sw)} samples")

    return k_fluid


def calculate_density_from_velocity(
    vp: Union[np.ndarray, float],
    vs: Optional[Union[np.ndarray, float]] = None,
    method: str = "gardner",
) -> np.ndarray:
    """Estimate density from velocity using empirical relationships.

    Args:
        vp: P-wave velocity (m/s).
        vs: S-wave velocity (m/s). If provided, uses more accurate method.
        method: Method to use: "gardner", "nafe_drake", or "brocher" (default "gardner").

    Returns:
        Estimated density (g/cc).

    Example:
        >>> from geosmith.primitives.petrophysics import calculate_density_from_velocity
        >>>
        >>> rho = calculate_density_from_velocity(vp=3000.0, method="gardner")
        >>> print(f"Density: {rho:.2f} g/cc")
    """
    vp = np.asarray(vp, dtype=float)
    vp_km_s = vp / 1000.0

    methods = {
        "gardner": lambda v: 1.74 * (v**0.25),
        "nafe_drake": lambda v: 1.5 + 0.5 * v,
        "brocher": lambda v: (
            1.6612 * v
            - 0.4721 * (v**2)
            + 0.0671 * (v**3)
            - 0.0043 * (v**4)
            + 0.000106 * (v**5)
        ),
    }

    if method not in methods:
        raise ValueError(
            f"Unknown method: {method}. Choose: {', '.join(methods.keys())}"
        )

    rho = methods[method](vp_km_s)

    # Clip to reasonable range
    rho = np.clip(rho, 1.0, 3.5)

    logger.debug(f"Estimated density using {method} method for {len(vp)} samples")

    return rho



def calculate_velocities_from_slowness(
    dtc: Union[np.ndarray, float],
    dts: Union[np.ndarray, float],
    units: str = "m/s",
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate P-wave and S-wave velocities from slowness (dtc, dts).

    Converts slowness (μs/ft or μs/m) to velocity (m/s).

    Args:
        dtc: P-wave slowness (compressional, typically in μs/ft or μs/m).
        dts: S-wave slowness (shear, typically in μs/ft or μs/m).
        units: Input units, 'm/s' (assumes input in μs/m, converts to m/s)
               or 'ft/s' (assumes input in μs/ft, converts to ft/s then to m/s).

    Returns:
        Tuple of (VP, VS) arrays in m/s.

    Example:
        >>> from geosmith.primitives.petrophysics import calculate_velocities_from_slowness
        >>>
        >>> dtc = np.array([100, 120, 140])  # μs/ft
        >>> dts = np.array([180, 200, 220])  # μs/ft
        >>> vp, vs = calculate_velocities_from_slowness(dtc, dts, units='ft/s')
    """
    dtc = np.asarray(dtc, dtype=np.float64)
    dts = np.asarray(dts, dtype=np.float64)

    if units == "m/s":
        # Input in μs/m, convert to m/s: v = 1e6 / dt
        vp = 1e6 / dtc
        vs = 1e6 / dts
    elif units == "ft/s":
        # Input in μs/ft, convert to ft/s, then to m/s
        vp_ft_s = 1e6 / dtc
        vs_ft_s = 1e6 / dts
        # Convert ft/s to m/s
        vp = vp_ft_s / 3.281
        vs = vs_ft_s / 3.281
    else:
        raise ValueError(f"Unknown units: {units}. Must be 'm/s' or 'ft/s'")

    # Handle division by zero
    vp = np.where(np.isinf(vp) | (dtc <= 0), np.nan, vp)
    vs = np.where(np.isinf(vs) | (dts <= 0), np.nan, vs)

    return vp, vs


def preprocess_avo_inputs(
    vp: Union[np.ndarray, float],
    vs: Union[np.ndarray, float],
    rho: Union[np.ndarray, float],
) -> dict[str, np.ndarray]:
    """Preprocess velocity and density data for AVO calculations.

    Calculates average values and differences between consecutive samples,
    which are required for AVO attribute calculations.

    Args:
        vp: P-wave velocity (m/s).
        vs: S-wave velocity (m/s).
        rho: Density (g/cc or kg/m³).

    Returns:
        Dictionary with keys:
            - 'VP_AVG': Average P-wave velocity between consecutive samples
            - 'VS_AVG': Average S-wave velocity between consecutive samples
            - 'RHO': Average density between consecutive samples
            - 'dVp': Difference in P-wave velocity between consecutive samples
            - 'dVs': Difference in S-wave velocity between consecutive samples
            - 'dRho': Difference in density between consecutive samples

    Example:
        >>> from geosmith.primitives.petrophysics import preprocess_avo_inputs
        >>>
        >>> vp = np.array([3000, 3100, 3200])
        >>> vs = np.array([1500, 1550, 1600])
        >>> rho = np.array([2.3, 2.4, 2.5])
        >>> preprocessed = preprocess_avo_inputs(vp, vs, rho)
    """
    vp = np.asarray(vp, dtype=np.float64)
    vs = np.asarray(vs, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)

    n = len(vp)

    if len(vs) != n or len(rho) != n:
        raise ValueError("vp, vs, and rho must have the same length")

    if n < 2:
        raise ValueError("Input arrays must have at least 2 samples")

    # Calculate differences
    dvp = np.zeros(n, dtype=np.float64)
    dvs = np.zeros(n, dtype=np.float64)
    drho = np.zeros(n, dtype=np.float64)

    # Calculate averages
    vp_avg = np.zeros(n, dtype=np.float64)
    vs_avg = np.zeros(n, dtype=np.float64)
    rho_avg = np.zeros(n, dtype=np.float64)

    for i in range(n - 1):
        dvp[i] = vp[i + 1] - vp[i]
        dvs[i] = vs[i + 1] - vs[i]
        drho[i] = rho[i + 1] - rho[i]

        vp_avg[i] = (vp[i] + vp[i + 1]) / 2.0
        vs_avg[i] = (vs[i] + vs[i + 1]) / 2.0
        rho_avg[i] = (rho[i] + rho[i + 1]) / 2.0

    # Use last calculated values for final sample
    dvp[-1] = dvp[-2]
    dvs[-1] = dvs[-2]
    drho[-1] = drho[-2]
    vp_avg[-1] = vp_avg[-2]
    vs_avg[-1] = vs_avg[-2]
    rho_avg[-1] = rho_avg[-2]

    return {
        "VP_AVG": vp_avg,
        "VS_AVG": vs_avg,
        "RHO": rho_avg,
        "dVp": dvp,
        "dVs": dvs,
        "dRho": drho,
    }


def calculate_avo_attributes(
    vp: Union[np.ndarray, float],
    vs: Union[np.ndarray, float],
    rho: Union[np.ndarray, float],
    return_all: bool = True,
) -> dict[str, np.ndarray]:
    """Calculate AVO (Amplitude Versus Offset) attributes from velocities and density.

    Computes AVO attributes including intercept, gradient, curvature, Poisson's ratio,
    reflectivities, and fluid factor based on Shuey (1985) approximation of Zoeppritz equations.

    Args:
        vp: P-wave velocity (m/s).
        vs: S-wave velocity (m/s).
        rho: Density (g/cc).
        return_all: If True, return dict with all attributes.
                   If False, return dict with only key attributes (A, B, PR, Rp, Rs, FF).

    Returns:
        Dictionary with AVO attributes:
            - 'k': Shear modulus ratio (VS/VP)^2
            - 'A': Intercept (zero-offset reflectivity)
            - 'B': Gradient
            - 'C': Curvature
            - 'productAB': A * B
            - 'AsignB': A * sign(B)
            - 'BsignA': B * sign(A)
            - 'PR': Poisson's Ratio
            - 'Rp': P-wave reflectivity
            - 'Rs': S-wave reflectivity
            - 'FF': Fluid Factor

    Example:
        >>> from geosmith.primitives.petrophysics import calculate_avo_attributes
        >>>
        >>> vp = np.array([3000, 3100, 3200])
        >>> vs = np.array([1500, 1550, 1600])
        >>> rho = np.array([2.3, 2.4, 2.5])
        >>> avo_dict = calculate_avo_attributes(vp, vs, rho)
        >>> print(f"Intercept: {avo_dict['A']}")
        >>> print(f"Gradient: {avo_dict['B']}")
    """
    vp = np.asarray(vp, dtype=np.float64)
    vs = np.asarray(vs, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)

    # Preprocess inputs
    preprocessed = preprocess_avo_inputs(vp, vs, rho)

    vp_avg = preprocessed["VP_AVG"]
    vs_avg = preprocessed["VS_AVG"]
    rho_avg = preprocessed["RHO"]
    dvp = preprocessed["dVp"]
    dvs = preprocessed["dVs"]
    drho = preprocessed["dRho"]

    # Shear modulus ratio: k = (VS/VP)^2
    k = (vs_avg / vp_avg) ** 2
    k = np.where(np.isnan(k) | np.isinf(k), 0, k)

    # Intercept (A): Zero-offset P-wave reflectivity
    # A = 0.5 * (dVp/VP_AVG + dRho/RHO)
    A = 0.5 * ((dvp / vp_avg) + (drho / rho_avg))

    # Gradient (B)
    # B = 0.5*(dVp/VP_AVG) - 2*k*(2*(dVs/VS_AVG) + dRho/RHO)
    B = 0.5 * (dvp / vp_avg) - (2 * k) * ((2 * (dvs / vs_avg)) + (drho / rho_avg))

    # Curvature (C)
    C = 0.5 * (dvp / vp_avg)

    # Product attributes
    productAB = A * B
    AsignB = A * np.sign(B)
    BsignA = B * np.sign(A)

    # Poisson's Ratio: PR = (gamma^2 - 2) / (2*gamma^2 - 2)
    # where gamma = VP/VS
    gamma = vp_avg / vs_avg
    gamma_sq = gamma ** 2
    PR = (gamma_sq - 2) / (2 * gamma_sq - 2)
    PR = np.where(np.isnan(PR) | np.isinf(PR), np.nan, PR)

    # P-wave Reflectivity (Rp)
    Rp = A.copy()

    # S-wave Reflectivity (Rs)
    Rs = 0.5 * (A - B)

    # Fluid Factor (Fatti et al., 1994)
    # FF = Rp - 1.16 * (VS_AVG/VP_AVG) * Rs
    vs_vp_ratio = vs_avg / vp_avg
    FF = Rp - (1.16 * vs_vp_ratio * Rs)

    # Create result dictionary
    result_dict = {
        "k": k,
        "A": A,
        "B": B,
        "C": C,
        "productAB": productAB,
        "AsignB": AsignB,
        "BsignA": BsignA,
        "PR": PR,
        "Rp": Rp,
        "Rs": Rs,
        "FF": FF,
    }

    if return_all:
        return result_dict
    else:
        # Return only key attributes
        return {
            "A": A,
            "B": B,
            "PR": PR,
            "Rp": Rp,
            "Rs": Rs,
            "FF": FF,
        }


def calculate_avo_from_slowness(
    dtc: Union[np.ndarray, float],
    dts: Union[np.ndarray, float],
    rho: Union[np.ndarray, float],
    units: str = "ft/s",
    return_all: bool = True,
) -> dict[str, np.ndarray]:
    """Calculate AVO attributes directly from slowness (dtc, dts) and density.

    Convenience function that combines velocity calculation and AVO attribute
    calculation in one step.

    Args:
        dtc: P-wave slowness (typically μs/ft or μs/m).
        dts: S-wave slowness (typically μs/ft or μs/m).
        rho: Density (g/cc).
        units: Input units for slowness, 'ft/s' (default) or 'm/s'.
        return_all: If True, return dict with all attributes.
                   If False, return dict with only key attributes.

    Returns:
        Dictionary with AVO attributes (see calculate_avo_attributes).

    Example:
        >>> from geosmith.primitives.petrophysics import calculate_avo_from_slowness
        >>>
        >>> dtc = np.array([100, 120, 140])  # μs/ft
        >>> dts = np.array([180, 200, 220])  # μs/ft
        >>> rho = np.array([2.3, 2.4, 2.5])  # g/cc
        >>> avo_dict = calculate_avo_from_slowness(dtc, dts, rho, units='ft/s')
    """
    vp, vs = calculate_velocities_from_slowness(dtc, dts, units=units)
    return calculate_avo_attributes(vp, vs, rho, return_all=return_all)



def calculate_permeability_wyllie_rose(
    phi: Union[np.ndarray, float],
    sw: Union[np.ndarray, float],
    coefficient: float = 0.625,
    porosity_exponent: float = 6.0,
    saturation_exponent: float = 2.0,
) -> np.ndarray:
    """Calculate permeability using Wyllie-Rose equation.

    k = C * (phi^a) / (sw^b)

    Similar to Timur but with different coefficients.
    Default coefficients from Wyllie & Rose (1950).

    Args:
        phi: Porosity (fraction, 0-1).
        sw: Water saturation (fraction, 0-1).
        coefficient: Coefficient C (default 0.625).
        porosity_exponent: Exponent a (default 6.0).
        saturation_exponent: Exponent b (default 2.0).

    Returns:
        Permeability in millidarcies (mD).

    Example:
        >>> from geosmith.primitives.petrophysics import calculate_permeability_wyllie_rose
        >>>
        >>> k = calculate_permeability_wyllie_rose(phi=0.25, sw=0.5)
        >>> print(f"Permeability: {k:.2f} mD")
    """
    phi = np.asarray(phi, dtype=float)
    sw = np.asarray(sw, dtype=float)

    if phi.size == 0 or sw.size == 0:
        raise ValueError(
            "Porosity and water saturation arrays must not be empty"
        )

    if phi.size != sw.size:
        raise ValueError(
            "Porosity and water saturation arrays must have same length"
        )

    phi = np.clip(phi, 0.01, 0.99)
    sw = np.clip(sw, 0.01, 0.99)

    k_md = coefficient * (phi**porosity_exponent) / (sw**saturation_exponent)

    k_md = np.clip(k_md, 0.001, 1e6)

    return k_md


def calculate_permeability_coates_dumanoir(
    phi: Union[np.ndarray, float],
    sw: Union[np.ndarray, float],
    coefficient: float = 70.0,
    porosity_exponent: float = 2.0,
    saturation_exponent: float = 2.0,
) -> np.ndarray:
    """Calculate permeability using Coates-Dumanoir equation.

    k = C * (phi^a) * ((1 - sw) / sw)^b

    Where irreducible water saturation is accounted for.
    Default coefficients from Coates & Dumanoir (1973).

    Args:
        phi: Porosity (fraction, 0-1).
        sw: Water saturation (fraction, 0-1).
        coefficient: Coefficient C (default 70.0).
        porosity_exponent: Exponent a (default 2.0).
        saturation_exponent: Exponent b (default 2.0).

    Returns:
        Permeability in millidarcies (mD).

    Example:
        >>> from geosmith.primitives.petrophysics import calculate_permeability_coates_dumanoir
        >>>
        >>> k = calculate_permeability_coates_dumanoir(phi=0.25, sw=0.5)
        >>> print(f"Permeability: {k:.2f} mD")
    """
    phi = np.asarray(phi, dtype=float)
    sw = np.asarray(sw, dtype=float)

    if phi.size == 0 or sw.size == 0:
        raise ValueError(
            "Porosity and water saturation arrays must not be empty"
        )

    if phi.size != sw.size:
        raise ValueError(
            "Porosity and water saturation arrays must have same length"
        )

    phi = np.clip(phi, 0.01, 0.99)
    sw = np.clip(sw, 0.01, 0.99)

    k_md = (
        coefficient
        * (phi**porosity_exponent)
        * ((1 - sw) / sw) ** saturation_exponent
    )

    k_md = np.clip(k_md, 0.001, 1e6)

    return k_md


def calculate_permeability_tixier(
    phi: Union[np.ndarray, float],
    sw: Union[np.ndarray, float],
    coefficient: float = 0.136,
    porosity_exponent: float = 4.4,
    saturation_exponent: float = 2.0,
) -> np.ndarray:
    """Calculate permeability using Tixier equation.

    k = C * (phi^a) / (sw^b)

    Similar to Timur equation. Default coefficients from Tixier (1949).

    Args:
        phi: Porosity (fraction, 0-1).
        sw: Water saturation (fraction, 0-1).
        coefficient: Coefficient C (default 0.136).
        porosity_exponent: Exponent a (default 4.4).
        saturation_exponent: Exponent b (default 2.0).

    Returns:
        Permeability in millidarcies (mD).

    Example:
        >>> from geosmith.primitives.petrophysics import calculate_permeability_tixier
        >>>
        >>> k = calculate_permeability_tixier(phi=0.25, sw=0.5)
        >>> print(f"Permeability: {k:.2f} mD")
    """
    # Tixier uses same form as Timur, just different default coefficients
    return calculate_permeability_timur(
        phi, sw, coefficient, porosity_exponent, saturation_exponent
    )


def calculate_porosity_from_density(
    rhob: Union[np.ndarray, float],
    rho_matrix: float = 2.65,
    rho_fluid: float = 1.0,
) -> np.ndarray:
    """Calculate porosity from bulk density.

    phi = (rho_matrix - rhob) / (rho_matrix - rho_fluid)

    Args:
        rhob: Bulk density (g/cc).
        rho_matrix: Matrix density (g/cc), default 2.65 (quartz).
        rho_fluid: Fluid density (g/cc), default 1.0 (water).

    Returns:
        Porosity (fraction, 0-1) as numpy array.

    Example:
        >>> from geosmith.primitives.petrophysics import calculate_porosity_from_density
        >>>
        >>> phi = calculate_porosity_from_density(rhob=2.4, rho_matrix=2.65, rho_fluid=1.0)
        >>> print(f"Porosity: {phi:.2f}")
    """
    rhob = np.asarray(rhob, dtype=float)

    if rhob.size == 0:
        raise ValueError("Bulk density array must not be empty")

    denominator = rho_matrix - rho_fluid
    if abs(denominator) < 1e-10:
        raise ValueError(
            f"Matrix and fluid densities too similar: {rho_matrix} vs {rho_fluid}"
        )

    phi = (rho_matrix - rhob) / denominator
    phi = np.clip(phi, 0, 1)

    return phi


def calculate_formation_factor(
    phi: Union[np.ndarray, float],
    m: float = 2.0,
    a: float = 1.0,
) -> np.ndarray:
    """Calculate formation resistivity factor.

    F = a / phi^m

    Args:
        phi: Porosity (fraction, 0-1).
        m: Cementation exponent (default 2.0).
        a: Tortuosity factor (default 1.0).

    Returns:
        Formation factor as numpy array.

    Example:
        >>> from geosmith.primitives.petrophysics import calculate_formation_factor
        >>>
        >>> F = calculate_formation_factor(phi=0.25, m=2.0, a=1.0)
        >>> print(f"Formation factor: {F:.2f}")
    """
    phi = np.asarray(phi, dtype=float)

    if phi.size == 0:
        raise ValueError("Porosity array must not be empty")

    phi = np.where(phi <= 0, np.nan, phi)
    F = a / (phi**m)

    return F

