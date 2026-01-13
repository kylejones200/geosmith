"""RasterGrid: Gridded values with affine transform, resolution, nodata."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from geosmith.objects.geoindex import GeoIndex


@dataclass(frozen=True)
class RasterGrid:
    """Immutable representation of a raster grid.

    Attributes:
        data: Array of shape (n_bands, n_rows, n_cols) or (n_rows, n_cols).
        transform: Affine transform as 6-element tuple (a, b, c, d, e, f).
        nodata: Optional nodata value.
        band_names: Optional list of band names, length must match n_bands.
        index: Optional GeoIndex with CRS and bounds information.
    """

    data: np.ndarray
    transform: Tuple[float, float, float, float, float, float]
    nodata: Optional[float] = None
    band_names: Optional[List[str]] = None
    index: Optional[GeoIndex] = None

    def __post_init__(self) -> None:
        """Validate RasterGrid parameters."""
        data = self.data
        if not isinstance(data, np.ndarray):
            raise ValueError(f"data must be numpy array, got {type(data)}")

        if data.ndim not in (2, 3):
            raise ValueError(f"data must be 2D or 3D array, got shape {data.shape}")

        if len(self.transform) != 6:
            raise ValueError(
                f"transform must have 6 elements, got {len(self.transform)}"
            )

        # Determine number of bands
        if data.ndim == 2:
            n_bands = 1
        else:
            n_bands = data.shape[0]

        if self.band_names is not None:
            if not isinstance(self.band_names, list):
                raise ValueError(
                    f"band_names must be list, got {type(self.band_names)}"
                )
            if len(self.band_names) != n_bands:
                raise ValueError(
                    f"band_names length ({len(self.band_names)}) must match "
                    f"number of bands ({n_bands})"
                )

        if self.index is not None:
            if not isinstance(self.index, GeoIndex):
                raise ValueError(f"index must be GeoIndex, got {type(self.index)}")

    def __repr__(self) -> str:
        """String representation."""
        if self.data.ndim == 2:
            n_bands = 1
            n_rows, n_cols = self.data.shape
        else:
            n_bands, n_rows, n_cols = self.data.shape
        has_nodata = self.nodata is not None
        has_names = self.band_names is not None
        has_index = self.index is not None
        return (
            f"RasterGrid(n_bands={n_bands}, shape=({n_rows}, {n_cols}), "
            f"has_nodata={has_nodata}, has_names={has_names}, has_index={has_index})"
        )
