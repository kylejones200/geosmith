"""GeoTable: Pandas DataFrame with geometry column."""

from dataclasses import dataclass
from typing import Optional, Union

import pandas as pd

from geosmith.objects.geoindex import GeoIndex
from geosmith.objects.lineset import LineSet
from geosmith.objects.pointset import PointSet
from geosmith.objects.polygonset import PolygonSet


@dataclass(frozen=True)
class GeoTable:
    """Immutable wrapper for pandas DataFrame with geometry.

    The geometry column contains lightweight geometry objects (PointSet,
    LineSet, or PolygonSet).

    Attributes:
        data: Pandas DataFrame with a geometry column.
        geometry_column: Name of the geometry column. Defaults to 'geometry'.
        index: Optional GeoIndex with CRS and bounds information.
    """

    data: pd.DataFrame
    geometry_column: str = "geometry"
    index: Optional[GeoIndex] = None

    def __post_init__(self) -> None:
        """Validate GeoTable parameters."""
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError(f"data must be pandas DataFrame, got {type(self.data)}")

        if self.geometry_column not in self.data.columns:
            raise ValueError(
                f"geometry column '{self.geometry_column}' not found in DataFrame. "
                f"Available columns: {list(self.data.columns)}"
            )

        # Validate geometry column contains valid geometry objects
        geom_col = self.data[self.geometry_column]
        for idx, geom in enumerate(geom_col):
            if not isinstance(geom, (PointSet, LineSet, PolygonSet)):
                raise ValueError(
                    f"geometry at index {idx} must be PointSet, LineSet, "
                    f"or PolygonSet, "
                    f"got {type(geom)}"
                )

        if self.index is not None:
            if not isinstance(self.index, GeoIndex):
                raise ValueError(f"index must be GeoIndex, got {type(self.index)}")

    def __repr__(self) -> str:
        """String representation."""
        n_rows = len(self.data)
        n_cols = len(self.data.columns)
        has_index = self.index is not None
        return (
            f"GeoTable(n_rows={n_rows}, n_cols={n_cols}, "
            f"geometry_column='{self.geometry_column}', has_index={has_index})"
        )
