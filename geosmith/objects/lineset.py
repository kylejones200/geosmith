"""LineSet: Lines with ordered vertices and optional attributes."""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from geosmith.objects.geoindex import GeoIndex


@dataclass(frozen=True)
class LineSet:
    """Immutable representation of a set of lines.

    Attributes:
        vertices: List of arrays, each array is shape (n_vertices, n_dims).
        attributes: Optional DataFrame with n_lines rows and attribute columns.
        index: Optional GeoIndex with CRS and bounds information.
    """

    vertices: List[np.ndarray]
    attributes: Optional[pd.DataFrame] = None
    index: Optional[GeoIndex] = None

    def __post_init__(self) -> None:
        """Validate LineSet parameters."""
        if not isinstance(self.vertices, list):
            raise ValueError(f"vertices must be list, got {type(self.vertices)}")

        if len(self.vertices) == 0:
            raise ValueError("vertices must have at least one line")

        n_dims = None
        for i, verts in enumerate(self.vertices):
            if not isinstance(verts, np.ndarray):
                raise ValueError(f"vertices[{i}] must be numpy array, got {type(verts)}")
            if verts.ndim != 2:
                raise ValueError(
                    f"vertices[{i}] must be 2D array, got shape {verts.shape}"
                )
            n_verts, dims = verts.shape
            if n_verts < 2:
                raise ValueError(
                    f"vertices[{i}] must have at least 2 vertices, got {n_verts}"
                )
            if dims < 2 or dims > 3:
                raise ValueError(
                    f"vertices[{i}] must have 2 or 3 dimensions, got {dims}"
                )
            if n_dims is None:
                n_dims = dims
            elif n_dims != dims:
                raise ValueError(
                    f"All lines must have same number of dimensions, "
                    f"got {n_dims} and {dims}"
                )

        n_lines = len(self.vertices)

        if self.attributes is not None:
            if not isinstance(self.attributes, pd.DataFrame):
                raise ValueError(
                    f"attributes must be pandas DataFrame, got {type(self.attributes)}"
                )
            if len(self.attributes) != n_lines:
                raise ValueError(
                    f"attributes length ({len(self.attributes)}) must match "
                    f"number of lines ({n_lines})"
                )

        if self.index is not None:
            if not isinstance(self.index, GeoIndex):
                raise ValueError(f"index must be GeoIndex, got {type(self.index)}")

    def __repr__(self) -> str:
        """String representation."""
        n_lines = len(self.vertices)
        n_dims = self.vertices[0].shape[1] if self.vertices else 0
        has_attrs = self.attributes is not None
        has_index = self.index is not None
        return (
            f"LineSet(n_lines={n_lines}, n_dims={n_dims}, "
            f"has_attributes={has_attrs}, has_index={has_index})"
        )

