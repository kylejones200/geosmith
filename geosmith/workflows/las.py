"""LAS (Log ASCII Standard) file loader.

Migrated from geosuite.io.las_loader.
Layer 4: Workflows - I/O operations.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

# Try to import lasio
try:
    import lasio

    LASIO_AVAILABLE = True
except ImportError:
    LASIO_AVAILABLE = False
    lasio = None


def read_las(
    las_path: Union[str, Path],
    version: Optional[str] = None,
) -> pd.DataFrame:
    """Load LAS file (supports LAS 2.0 and LAS 3.0).

    Args:
        las_path: Path to LAS file.
        version: LAS version ('2.0' or '3.0'). Auto-detected if not specified.

    Returns:
        Well log data with depth/index as first column.

    Raises:
        ImportError: If lasio is not installed.
        FileNotFoundError: If LAS file doesn't exist.
        ValueError: If file cannot be read.

    Example:
        >>> from geosmith.workflows.las import read_las
        >>> df = read_las('well_log.las')
        >>> print(df.columns)
    """
    if not LASIO_AVAILABLE:
        raise ImportError(
            "lasio is required for LAS support. " "Install with: pip install lasio"
        )

    las_path = Path(las_path)
    if not las_path.exists():
        raise FileNotFoundError(f"LAS file not found: {las_path}")

    # Read LAS file
    try:
        las = lasio.read(str(las_path))
    except Exception as e:
        raise ValueError(f"Error reading LAS file: {e}") from e

    # Detect version
    detected_version = _detect_las_version(las)
    if version and version != detected_version:
        logger.warning(
            f"Specified version {version} differs from detected {detected_version}"
        )

    # Convert to DataFrame
    df = las.df()

    # Handle LAS 3.0 specific features
    if detected_version == "3.0":
        df = _process_las3_features(las, df)

    # Ensure depth column is properly named
    if df.index.name is None or df.index.name == "":
        df.index.name = "DEPTH"

    # Reset index to make depth a column
    df = df.reset_index()

    logger.info(
        f"Loaded LAS {detected_version} file: {len(df)} rows, "
        f"{len(df.columns)} columns"
    )

    return df


def _detect_las_version(las) -> str:
    """Detect LAS file version.

    Args:
        las: lasio.LASFile object.

    Returns:
        Version string ('2.0' or '3.0').
    """
    # Check version in header
    if hasattr(las, "version"):
        version_str = str(las.version)
        if "3.0" in version_str:
            return "3.0"

    # Check for LAS 3.0 specific features
    if hasattr(las, "well") and hasattr(las.well, "metadata"):
        return "3.0"

    # Default to 2.0 for legacy files
    return "2.0"


def _process_las3_features(las, df: pd.DataFrame) -> pd.DataFrame:
    """Process LAS 3.0 specific features.

    Args:
        las: lasio.LASFile object.
        df: DataFrame from LAS file.

    Returns:
        Processed DataFrame with LAS 3.0 enhancements.
    """
    # LAS 3.0 has better metadata handling
    if hasattr(las, "well"):
        well_metadata = {}
        for item in las.well:
            if hasattr(item, "mnemonic") and hasattr(item, "value"):
                well_metadata[item.mnemonic] = item.value

        # Store metadata as DataFrame attribute
        df.attrs["well_metadata"] = well_metadata

    return df
