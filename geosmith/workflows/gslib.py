"""GSLIB format I/O for geostatistical data.

GSLIB (Geostatistical Software Library) is a standard format for geostatistics.
This module provides functions to export GeoSmith data to GSLIB format for
compatibility with industry-standard software.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from geosmith.objects.pointset import PointSet
from geosmith.objects.rastergrid import RasterGrid


def write_gslib(
    data: Union[PointSet, RasterGrid, np.ndarray, pd.DataFrame],
    filename: Union[str, Path],
    variable_names: Optional[list[str]] = None,
    title: str = "GeoSmith Export",
) -> None:
    """Write data to GSLIB format.

    GSLIB format consists of:
    1. Title line
    2. Number of variables
    3. Variable names (one per line)
    4. Data (space-separated values)

    Args:
        data: Data to export (PointSet, RasterGrid, array, or DataFrame).
        filename: Output filename.
        variable_names: Optional list of variable names.
        title: Title for GSLIB file (default: "GeoSmith Export").

    Example:
        >>> from geosmith import PointSet
        >>> from geosmith.workflows.gslib import write_gslib
        >>>
        >>> points = PointSet(coordinates=coords)
        >>> write_gslib(points, "data.dat", variable_names=["X", "Y", "Z", "Grade"])
    """
    filename = Path(filename)

    # Convert to DataFrame format
    if isinstance(data, PointSet):
        df = _pointset_to_dataframe(data, variable_names)
    elif isinstance(data, RasterGrid):
        df = _rastergrid_to_dataframe(data, variable_names)
    elif isinstance(data, np.ndarray):
        df = pd.DataFrame(data)
        if variable_names:
            df.columns = variable_names[: data.shape[1]]
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

    # Write GSLIB file
    with open(filename, "w") as f:
        # Title
        f.write(f"{title}\n")

        # Number of variables
        n_vars = len(df.columns)
        f.write(f"{n_vars}\n")

        # Variable names
        for col in df.columns:
            f.write(f"{col}\n")

        # Data
        for _, row in df.iterrows():
            values = " ".join(f"{val:.6f}" if pd.notna(val) else "-999.0" for val in row)
            f.write(f"{values}\n")


def read_gslib(
    filename: Union[str, Path],
) -> tuple[pd.DataFrame, str]:
    """Read data from GSLIB format.

    Args:
        filename: Input filename.

    Returns:
        Tuple of (DataFrame, title).
    """
    filename = Path(filename)

    with open(filename, "r") as f:
        # Read title
        title = f.readline().strip()

        # Read number of variables
        n_vars = int(f.readline().strip())

        # Read variable names
        variable_names = [f.readline().strip() for _ in range(n_vars)]

        # Read data
        data = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            values = [float(x) if x != "-999.0" else np.nan for x in line.split()]
            data.append(values)

    df = pd.DataFrame(data, columns=variable_names)
    return df, title


def _pointset_to_dataframe(
    points: PointSet, variable_names: Optional[list[str]] = None
) -> pd.DataFrame:
    """Convert PointSet to DataFrame for GSLIB export."""
    coords = points.coordinates
    n_dims = coords.shape[1]

    # Default coordinate names
    if variable_names is None:
        coord_names = ["X", "Y", "Z"][:n_dims]
        if points.attributes is not None:
            attr_names = list(points.attributes.columns)
            variable_names = coord_names + attr_names
        else:
            variable_names = coord_names
    else:
        if len(variable_names) < n_dims:
            raise ValueError(
                f"variable_names must have at least {n_dims} elements for coordinates"
            )

    # Create DataFrame
    data_dict = {}
    for i, name in enumerate(variable_names[:n_dims]):
        data_dict[name] = coords[:, i]

    # Add attributes if present
    if points.attributes is not None:
        for col in points.attributes.columns:
            data_dict[col] = points.attributes[col].values

    return pd.DataFrame(data_dict)


def _rastergrid_to_dataframe(
    grid: RasterGrid, variable_names: Optional[list[str]] = None
) -> pd.DataFrame:
    """Convert RasterGrid to DataFrame for GSLIB export."""
    # Get grid shape
    if grid.data.ndim == 2:
        n_bands = 1
        n_rows, n_cols = grid.data.shape
        data_2d = grid.data[np.newaxis, :, :]
    else:
        n_bands, n_rows, n_cols = grid.data.shape
        data_2d = grid.data

    # Get transform parameters
    a, b, c, d, e, f = grid.transform

    # Generate coordinates
    col_indices = np.arange(n_cols)
    row_indices = np.arange(n_rows)
    col_coords, row_coords = np.meshgrid(col_indices, row_indices)

    # Apply affine transform
    x_coords = a * col_coords + b * row_coords + c
    y_coords = d * col_coords + e * row_coords + f

    # Flatten
    x_flat = x_coords.ravel()
    y_flat = y_coords.ravel()

    # Default variable names
    if variable_names is None:
        if grid.band_names:
            variable_names = ["X", "Y"] + grid.band_names
        else:
            variable_names = ["X", "Y"] + [f"Band{i+1}" for i in range(n_bands)]
    else:
        if len(variable_names) < 2 + n_bands:
            raise ValueError(
                f"variable_names must have at least {2 + n_bands} elements "
                f"(X, Y, and {n_bands} bands)"
            )

    # Create DataFrame
    data_dict = {
        variable_names[0]: x_flat,
        variable_names[1]: y_flat,
    }

    for i in range(n_bands):
        band_data = data_2d[i, :, :].ravel()
        # Handle nodata
        if grid.nodata is not None:
            band_data = np.where(band_data == grid.nodata, np.nan, band_data)
        data_dict[variable_names[2 + i]] = band_data

    return pd.DataFrame(data_dict)


def export_block_model_gslib(
    block_model: pd.DataFrame,
    filename: Union[str, Path],
    coordinate_cols: Optional[list[str]] = None,
    title: str = "Block Model Export",
) -> None:
    """Export block model to GSLIB format.

    Args:
        block_model: DataFrame with block model data.
        filename: Output filename.
        coordinate_cols: List of coordinate column names (default: ["X", "Y", "Z"]).
        title: Title for GSLIB file.
    """
    if coordinate_cols is None:
        coordinate_cols = ["X", "Y", "Z"]
        # Check if columns exist
        for col in coordinate_cols:
            if col not in block_model.columns:
                raise ValueError(f"Coordinate column '{col}' not found in block model")

    # Reorder columns: coordinates first, then attributes
    other_cols = [col for col in block_model.columns if col not in coordinate_cols]
    ordered_cols = coordinate_cols + other_cols
    df_ordered = block_model[ordered_cols]

    write_gslib(df_ordered, filename, variable_names=ordered_cols, title=title)

