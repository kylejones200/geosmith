"""Geosmith petrophysics: Rock physics calculations
(Gassmann, fluid properties, porosity)

Migrated from geosuite.petro.
Layer 2: Primitives - Pure operations.
"""

from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from geosmith.primitives.petrophysics._common import logger, njit


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

    K_sat_final = K_dry + (1 - K_dry/K_mineral)^2 /
        (phi/K_fluid_final + (1-phi)/K_mineral - K_dry/K_mineral^2)

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

    logger.debug(f"Performing Gassmann fluid substitution for {len(phi)} samples")

    # Gassmann's equation
    # K_sat = K_dry + (1 - K_dry/K_mineral)^2 /
    # (phi/K_fluid + (1-phi)/K_mineral - K_dry/K_mineral^2)
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
        method: Method to use: "gardner", "nafe_drake", or "brocher"
            (default "gardner").

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
        >>> from geosmith.primitives.petrophysics import (
        ...     calculate_velocities_from_slowness
        ... )
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
        >>> phi = calculate_porosity_from_density(
        ...     rhob=2.4, rho_matrix=2.65, rho_fluid=1.0
        ... )
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
