"""Geosmith petrophysics: Water saturation calculations (Archie, Simandoux, Indonesia, Waxman-Smits)

Migrated from geosuite.petro.
Layer 2: Primitives - Pure operations.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from geosmith.primitives.petrophysics._common import logger, njit


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
