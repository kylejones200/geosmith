"""PolygonSet: Polygons with rings and optional attributes."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from geosmith.objects.geoindex import GeoIndex


@dataclass(frozen=True)
class PolygonSet:
    """Immutable representation of a set of polygons.

    Each polygon is represented by a list of rings, where the first ring
    is the exterior ring and subsequent rings are holes.

    Attributes:
        rings: List of lists of arrays. Outer list has n_polygons elements.
            Each inner list has rings for that polygon. Each ring is an array
            of shape (n_vertices, n_dims).
        attributes: Optional DataFrame with n_polygons rows and attribute columns.
        index: Optional GeoIndex with CRS and bounds information.
    """

    rings: List[List[np.ndarray]]
    attributes: Optional[pd.DataFrame] = None
    index: Optional[GeoIndex] = None

    def __post_init__(self) -> None:
        """Validate PolygonSet parameters."""
        if not isinstance(self.rings, list):
            raise ValueError(f"rings must be list, got {type(self.rings)}")

        if len(self.rings) == 0:
            raise ValueError("rings must have at least one polygon")

        n_dims = None
        for poly_idx, polygon_rings in enumerate(self.rings):
            if not isinstance(polygon_rings, list):
                raise ValueError(
                    f"rings[{poly_idx}] must be list, got {type(polygon_rings)}"
                )
            if len(polygon_rings) == 0:
                raise ValueError(f"rings[{poly_idx}] must have at least one ring")

            for ring_idx, ring in enumerate(polygon_rings):
                if not isinstance(ring, np.ndarray):
                    raise ValueError(
                        f"rings[{poly_idx}][{ring_idx}] must be numpy array, "
                        f"got {type(ring)}"
                    )
                if ring.ndim != 2:
                    raise ValueError(
                        f"rings[{poly_idx}][{ring_idx}] must be 2D array, "
                        f"got shape {ring.shape}"
                    )
                n_verts, dims = ring.shape
                if n_verts < 3:
                    raise ValueError(
                        f"rings[{poly_idx}][{ring_idx}] must have at least 3 vertices, "
                        f"got {n_verts}"
                    )
                if dims < 2 or dims > 3:
                    raise ValueError(
                        f"rings[{poly_idx}][{ring_idx}] must have 2 or 3 dimensions, "
                        f"got {dims}"
                    )
                if n_dims is None:
                    n_dims = dims
                elif n_dims != dims:
                    raise ValueError(
                        f"All rings must have same number of dimensions, "
                        f"got {n_dims} and {dims}"
                    )

        n_polygons = len(self.rings)

        if self.attributes is not None:
            if not isinstance(self.attributes, pd.DataFrame):
                raise ValueError(
                    f"attributes must be pandas DataFrame, got {type(self.attributes)}"
                )
            if len(self.attributes) != n_polygons:
                raise ValueError(
                    f"attributes length ({len(self.attributes)}) must match "
                    f"number of polygons ({n_polygons})"
                )

        if self.index is not None:
            if not isinstance(self.index, GeoIndex):
                raise ValueError(f"index must be GeoIndex, got {type(self.index)}")

    def __repr__(self) -> str:
        """String representation."""
        n_polygons = len(self.rings)
        n_dims = (
            self.rings[0][0].shape[1] if self.rings and self.rings[0] else 0
        )
        has_attrs = self.attributes is not None
        has_index = self.index is not None
        return (
            f"PolygonSet(n_polygons={n_polygons}, n_dims={n_dims}, "
            f"has_attributes={has_attrs}, has_index={has_index})"
        )

