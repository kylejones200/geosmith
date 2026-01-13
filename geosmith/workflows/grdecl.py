"""GRDECL file I/O for reservoir modeling.

Migrated from pygeomodeling.grdecl_parser.
Layer 4: Workflows - I/O operations.
"""

import logging
import re
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from geosmith.objects.rastergrid import RasterGrid
from geosmith.objects.geoindex import GeoIndex

logger = logging.getLogger(__name__)


def read_grdecl(
    filepath: str | Path, property_name: Optional[str] = None
) -> Dict | RasterGrid:
    """Read GRDECL file and extract reservoir grid properties.

    Args:
        filepath: Path to GRDECL file.
        property_name: Optional property name to extract (e.g., 'PERMX').
                      If None, returns all properties as dict.

    Returns:
        If property_name specified: RasterGrid with the property.
        If property_name is None: Dictionary with 'dimensions' and 'properties'.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file format is invalid.

    Example:
        >>> from geosmith.workflows.grdecl import read_grdecl
        >>>
        >>> # Read all properties
        >>> data = read_grdecl('SPE9.GRDECL')
        >>> print(f"Grid: {data['dimensions']}")
        >>>
        >>> # Read specific property as RasterGrid
        >>> permx = read_grdecl('SPE9.GRDECL', property_name='PERMX')
        >>> print(f"Permeability grid: {permx.data.shape}")
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"GRDECL file not found: {filepath}")

    logger.info(f"Reading GRDECL file: {filepath}")

    try:
        with open(filepath) as f:
            content = f.read()
    except Exception as e:
        raise ValueError(f"Error reading GRDECL file: {e}") from e

    # Parse SPECGRID to get dimensions
    specgrid_match = re.search(
        r"SPECGRID\s*\n\s*(\d+)\s+(\d+)\s+(\d+)", content, re.IGNORECASE
    )
    if not specgrid_match:
        raise ValueError(
            "SPECGRID keyword not found in GRDECL file. "
            "This is required to define grid dimensions."
        )

    nx, ny, nz = map(int, specgrid_match.groups())
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError(f"Invalid grid dimensions: {nx} x {ny} x {nz}")

    total_cells = nx * ny * nz
    logger.info(f"Grid dimensions: {nx} x {ny} x {nz} = {total_cells} cells")

    # Parse properties
    properties_to_parse = ["PERMX", "PERMY", "PERMZ", "PORO", "NTG"]
    properties = {}

    for prop in properties_to_parse:
        try:
            prop_data = _parse_property(content, prop)
            if len(prop_data) == total_cells:
                # Reshape to 3D array (Fortran order for reservoir modeling)
                prop_3d = prop_data.reshape((nx, ny, nz), order="F")
                properties[prop] = prop_3d
                logger.info(f"Loaded {prop}: {len(prop_data)} values")
        except ValueError:
            logger.warning(f"Could not load {prop}")

    if not properties:
        raise ValueError(
            "No properties were successfully loaded from the GRDECL file. "
            f"Expected one of: {', '.join(properties_to_parse)}"
        )

    # If specific property requested, return as RasterGrid
    if property_name:
        if property_name not in properties:
            raise ValueError(
                f"Property '{property_name}' not found. "
                f"Available properties: {list(properties.keys())}"
            )

        data = properties[property_name]
        # Create simple transform (would need proper grid geometry in production)
        transform = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
        index = GeoIndex(
            crs=None,  # GRDECL doesn't specify CRS
            bounds=(0, 0, nx, ny),
        )

        return RasterGrid(
            data=data,
            transform=transform,
            index=index,
        )

    # Return all properties as dict
    return {"dimensions": (nx, ny, nz), "properties": properties}


def _parse_property(content: str, property_name: str) -> np.ndarray:
    """Parse a property section from GRDECL content.

    Args:
        content: GRDECL file content.
        property_name: Name of property to parse.

    Returns:
        Array of property values.

    Raises:
        ValueError: If property not found.
    """
    pattern = rf"{property_name}\s*\n(.*?)(?=\n[A-Z]|\n--|\Z)"
    match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)

    if not match:
        raise ValueError(f"Property {property_name} not found in file")

    property_data = match.group(1)

    # Extract numerical values, handling comments and forward slashes
    numbers = []
    for line in property_data.split("\n"):
        # Remove comments (lines starting with --)
        line = re.sub(r"--.*", "", line)
        # Remove trailing / and whitespace
        line = re.sub(r"/.*", "", line)
        # Extract numbers
        line_numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?", line)
        numbers.extend([float(x) for x in line_numbers])

    return np.array(numbers)


def write_grdecl(
    data: Dict | RasterGrid,
    filepath: str | Path,
    property_name: Optional[str] = None,
) -> None:
    """Write GRDECL file from reservoir grid data.

    Args:
        data: Dictionary with 'dimensions' and 'properties', or RasterGrid.
        filepath: Output file path.
        property_name: Optional property name if data is RasterGrid.

    Raises:
        ValueError: If data format is invalid.
    """
    filepath = Path(filepath)
    logger.info(f"Writing GRDECL file: {filepath}")

    if isinstance(data, RasterGrid):
        if property_name is None:
            raise ValueError("property_name must be specified when data is RasterGrid")

        # Extract dimensions from RasterGrid
        if data.data.ndim == 3:
            nz, ny, nx = data.data.shape
        else:
            raise ValueError("RasterGrid must be 3D for GRDECL export")

        properties = {property_name: data.data}
    elif isinstance(data, dict):
        if "dimensions" not in data or "properties" not in data:
            raise ValueError("Dictionary must have 'dimensions' and 'properties' keys")
        nx, ny, nz = data["dimensions"]
        properties = data["properties"]
    else:
        raise ValueError(f"data must be dict or RasterGrid, got {type(data)}")

    # Write GRDECL file
    with open(filepath, "w") as f:
        # Write SPECGRID
        f.write(f"SPECGRID\n{nx} {ny} {nz} /\n\n")

        # Write each property
        for prop_name, prop_data in properties.items():
            f.write(f"{prop_name}\n")
            # Flatten in Fortran order
            flat = prop_data.flatten(order="F")
            # Write in columns of 6 (GRDECL format)
            for i in range(0, len(flat), 6):
                chunk = flat[i : i + 6]
                f.write(" ".join(f"{val:.6E}" for val in chunk) + "\n")
            f.write("/\n\n")

    logger.info(f"Wrote GRDECL file with {len(properties)} properties")


def export_block_model(
    block_model: pd.DataFrame,
    filename: Union[str, Path],
    format: str = "csv",
    include_metadata: bool = True,
) -> None:
    """Export block model to file.

    Supports CSV format (compatible with Vulcan, Datamine, Leapfrog, Surpac).
    Standard columns include X, Y, Z coordinates and grade estimates.

    Args:
        block_model: DataFrame with block model data.
        filename: Output file path.
        format: Export format ('csv', 'parquet'). Default 'csv' for compatibility.
        include_metadata: If True, add comment lines with metadata.

    Raises:
        ValueError: If format is unsupported.

    Example:
        >>> block_model = pd.DataFrame({
        ...     'x': [100, 125, 150],
        ...     'y': [200, 200, 200],
        ...     'z': [50, 50, 50],
        ...     'grade': [2.5, 1.8, 2.1]
        ... })
        >>> export_block_model(block_model, 'block_model.csv')
    """
    filename = Path(filename)

    if format == "csv":
        # CSV export with optional metadata
        if include_metadata:
            with open(filename, "w") as f:
                f.write("# Block Model Export\n")
                f.write(f"# Total blocks: {len(block_model)}\n")
                f.write(f"# Columns: {', '.join(block_model.columns)}\n")
                f.write("#\n")
            # Append CSV data
            block_model.to_csv(filename, mode="a", index=False)
        else:
            block_model.to_csv(filename, index=False)

        logger.info(f"Exported block model to {filename} ({len(block_model):,} blocks)")

    elif format == "parquet":
        block_model.to_parquet(filename, index=False)
        logger.info(f"Exported block model to {filename} ({len(block_model):,} blocks)")

    else:
        raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'parquet'")
