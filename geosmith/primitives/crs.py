"""Coordinate Reference System (CRS) primitives.

Pure CRS operations that work with Layer 1 objects.
Migrated from geosuite.io.crs_utils.
Layer 2: Primitives - Pure operations.
"""

import logging
from typing import Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Try to import pyproj for CRS handling
try:
    from pyproj import CRS, Transformer

    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False
    CRS = None  # type: ignore
    Transformer = None  # type: ignore


def standardize_crs(crs: Union[str, int, "CRS"]) -> "CRS":
    """Standardize CRS to pyproj.CRS object.

    Args:
        crs: Coordinate reference system (EPSG code, WKT, PROJ string, or CRS object).

    Returns:
        Standardized CRS object.

    Raises:
        ImportError: If pyproj is not installed.
        ValueError: If CRS format is invalid.

    Example:
        >>> from geosmith.primitives.crs import standardize_crs
        >>> crs = standardize_crs('EPSG:4326')
        >>> print(crs.to_epsg())
    """
    if not PYPROJ_AVAILABLE:
        raise ImportError(
            "pyproj is required for CRS support. "
            "Install with: pip install pyproj"
        )

    if isinstance(crs, CRS):  # type: ignore
        return crs

    if isinstance(crs, int):
        return CRS.from_epsg(crs)  # type: ignore

    if isinstance(crs, str):
        if crs.startswith("EPSG:"):
            return CRS.from_epsg(int(crs.split(":")[1]))  # type: ignore
        elif crs.isdigit():
            return CRS.from_epsg(int(crs))  # type: ignore
        else:
            return CRS.from_string(crs)  # type: ignore

    raise ValueError(f"Invalid CRS format: {crs}")


def transform_coordinates(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    source_crs: Union[str, int, "CRS"],
    target_crs: Union[str, int, "CRS"],
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Transform coordinates from source CRS to target CRS.

    Args:
        x: X coordinates (float or array).
        y: Y coordinates (float or array).
        source_crs: Source coordinate reference system.
        target_crs: Target coordinate reference system.

    Returns:
        Tuple of (x_transformed, y_transformed).

    Raises:
        ImportError: If pyproj is not installed.

    Example:
        >>> from geosmith.primitives.crs import transform_coordinates
        >>> x_new, y_new = transform_coordinates(
        ...     x=500000, y=6000000,
        ...     source_crs='EPSG:32633',
        ...     target_crs='EPSG:4326'
        ... )
    """
    if not PYPROJ_AVAILABLE:
        raise ImportError(
            "pyproj is required for CRS transformation. "
            "Install with: pip install pyproj"
        )

    source_crs_obj = standardize_crs(source_crs)
    target_crs_obj = standardize_crs(target_crs)

    transformer = Transformer.from_crs(
        source_crs_obj, target_crs_obj, always_xy=True
    )

    # Convert to numpy arrays
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    # Transform
    x_transformed, y_transformed = transformer.transform(x_arr, y_arr)

    # Return in same format as input
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return float(x_transformed), float(y_transformed)
    else:
        return x_transformed, y_transformed


def get_epsg_code(crs: Union[str, int, "CRS"]) -> Optional[int]:
    """Get EPSG code from CRS if available.

    Args:
        crs: Coordinate reference system.

    Returns:
        EPSG code or None if not available.

    Example:
        >>> from geosmith.primitives.crs import get_epsg_code
        >>> epsg = get_epsg_code('EPSG:4326')
        >>> print(epsg)  # 4326
    """
    if not PYPROJ_AVAILABLE:
        return None

    try:
        crs_obj = standardize_crs(crs)
        return crs_obj.to_epsg()
    except Exception:
        return None


def validate_coordinates(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    bounds: Optional[Tuple[float, float, float, float]] = None,
) -> bool:
    """Validate coordinates are within expected bounds.

    Args:
        x: X coordinates.
        y: Y coordinates.
        bounds: Optional expected bounds (x_min, y_min, x_max, y_max).

    Returns:
        True if coordinates are valid.

    Example:
        >>> from geosmith.primitives.crs import validate_coordinates
        >>> is_valid = validate_coordinates(
        ...     x=[100, 200, 300],
        ...     y=[50, 60, 70],
        ...     bounds=(0, 0, 500, 500)
        ... )
    """
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    # Check for NaN or Inf
    if np.any(np.isnan(x_arr)) or np.any(np.isnan(y_arr)):
        return False
    if np.any(np.isinf(x_arr)) or np.any(np.isinf(y_arr)):
        return False

    # Check bounds if provided
    if bounds is not None:
        x_min, y_min, x_max, y_max = bounds
        if np.any(x_arr < x_min) or np.any(x_arr > x_max):
            return False
        if np.any(y_arr < y_min) or np.any(y_arr > y_max):
            return False

    return True


def get_common_crs() -> dict[str, int]:
    """Get dictionary of common CRS codes.

    Returns:
        Dictionary mapping CRS names to EPSG codes.

    Example:
        >>> from geosmith.primitives.crs import get_common_crs
        >>> crs_dict = get_common_crs()
        >>> print(crs_dict['WGS84'])  # 4326
    """
    return {
        "WGS84": 4326,
        "UTM Zone 33N": 32633,
        "UTM Zone 33S": 32733,
        "UTM Zone 15N": 32615,
        "UTM Zone 15S": 32715,
        "NAD83": 4269,
        "NAD27": 4267,
        "Web Mercator": 3857,
    }

