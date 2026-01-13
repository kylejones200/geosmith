"""Geomechanics: Fracture orientation and permeability

Pure geomechanics operations - fracture module.
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

# Optional scipy for von Mises distribution
try:
    from scipy.stats import vonmises

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


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

    fracture_strike = (
        np.where(
            is_normal,
            shmin_azimuth + 90.0,  # Perpendicular to Shmin
            np.where(
                is_strike_slip,
                shmax_azimuth + fracture_angle,  # At angle to Shmax
                shmax_azimuth,  # Parallel to Shmax for reverse
            ),
        )
        % 360.0
    )

    # Fracture dip depends on stress regime
    fracture_dip = np.where(
        is_normal,
        60.0,  # Steep dip for normal faulting
        np.where(
            is_strike_slip, 90.0, 30.0
        ),  # Vertical for strike-slip, shallow for reverse
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
    # Ensure arrays are 1D (handle scalar inputs)
    fracture_strike = np.atleast_1d(fracture_strike)
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
    # Ensure arrays are 1D (handle scalar inputs)
    fracture_strike = np.atleast_1d(fracture_strike)
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
        >>> from geosmith.primitives.geomechanics import (
        ...     fracture_orientation_distribution
        ... )
        >>>
        >>> strikes = fracture_orientation_distribution(
        ...     mean_strike=45.0, concentration=10.0, n_samples=1000
        ... )
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
        >>> aperture = calculate_fracture_aperture(
        ...     normal_stress=20.0, closure_stress=5.0
        ... )
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
