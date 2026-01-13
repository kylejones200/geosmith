"""PointSet: Point clouds with coordinates and optional attributes."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from geosmith.objects.geoindex import GeoIndex


@dataclass(frozen=True)
class PointSet:
    """Immutable representation of a point cloud.

    Attributes:
        coordinates: Array of shape (n_points, n_dims) with point coordinates.
        attributes: Optional DataFrame with n_points rows and attribute columns.
        index: Optional GeoIndex with CRS and bounds information.
    """

    coordinates: np.ndarray
    attributes: Optional[pd.DataFrame] = None
    index: Optional[GeoIndex] = None

    def __post_init__(self) -> None:
        """Validate PointSet parameters."""
        coords = self.coordinates
        if not isinstance(coords, np.ndarray):
            raise ValueError(f"coordinates must be numpy array, got {type(coords)}")

        if coords.ndim != 2:
            raise ValueError(f"coordinates must be 2D array, got shape {coords.shape}")

        n_points, n_dims = coords.shape
        if n_points == 0:
            raise ValueError("coordinates must have at least one point")

        if n_dims < 2 or n_dims > 3:
            raise ValueError(f"coordinates must have 2 or 3 dimensions, got {n_dims}")

        if self.attributes is not None:
            if not isinstance(self.attributes, pd.DataFrame):
                raise ValueError(
                    f"attributes must be pandas DataFrame, got {type(self.attributes)}"
                )
            if len(self.attributes) != n_points:
                raise ValueError(
                    f"attributes length ({len(self.attributes)}) must match "
                    f"number of points ({n_points})"
                )

        if self.index is not None:
            if not isinstance(self.index, GeoIndex):
                raise ValueError(f"index must be GeoIndex, got {type(self.index)}")

    def __repr__(self) -> str:
        """String representation."""
        n_points, n_dims = self.coordinates.shape
        has_attrs = self.attributes is not None
        has_index = self.index is not None
        return (
            f"PointSet(n_points={n_points}, n_dims={n_dims}, "
            f"has_attributes={has_attrs}, has_index={has_index})"
        )
