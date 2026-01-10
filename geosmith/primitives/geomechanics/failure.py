"""Geomechanics: Failure criteria calculations

Pure geomechanics operations - failure module.
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

