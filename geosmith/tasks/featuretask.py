"""FeatureTask: Vector feature creation operations."""

import logging
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from geosmith.objects.geoindex import GeoIndex
from geosmith.objects.geotable import GeoTable
from geosmith.objects.pointset import PointSet
from geosmith.objects.polygonset import PolygonSet
from geosmith.primitives.geometry import (
    bounding_box,
    nearest_neighbor_search,
    point_in_polygon,
)

logger = logging.getLogger(__name__)


class FeatureTask:
    """Task for vector feature creation operations.

    Supports buffers, joins, distances, and zonal summaries.
    """

    def __init__(self, crs: Optional[str] = None):
        """Initialize FeatureTask.

        Args:
            crs: Optional default CRS for output geometries.
        """
        self.crs = crs

    def buffer(
        self,
        geometry: Union[GeoTable, PolygonSet, PointSet],
        distance: float,
        **kwargs: Any,
    ) -> PolygonSet:
        """Create buffer around geometry.

        Args:
            geometry: Input geometry (GeoTable, PolygonSet, or PointSet).
            distance: Buffer distance.
            **kwargs: Additional arguments.

        Returns:
            PolygonSet with buffered geometries.
        """
        logger.info(f"Creating buffer with distance={distance}")

        # Convert input to objects
        if isinstance(geometry, GeoTable):
            # Extract geometries from GeoTable
            geoms = geometry.data[geometry.geometry_column].tolist()
            if isinstance(geoms[0], PointSet):
                points = geoms
            elif isinstance(geoms[0], PolygonSet):
                # Flatten polygons
                all_rings = []
                for poly in geoms:
                    all_rings.extend(poly.rings)
                geometry = PolygonSet(rings=all_rings, index=geometry.index)
            else:
                raise ValueError(
                    f"Unsupported geometry type in GeoTable: {type(geoms[0])}"
                )

        # Simple buffer implementation (would use shapely in production)
        if isinstance(geometry, PointSet):
            return self._buffer_points(geometry, distance)
        elif isinstance(geometry, PolygonSet):
            return self._buffer_polygons(geometry, distance)
        else:
            raise ValueError(f"Unsupported geometry type: {type(geometry)}")

    def _buffer_points(self, points: PointSet, distance: float) -> PolygonSet:
        """Buffer points by creating circles."""
        try:
            from geosmith.tasks._geopandas_adapter import buffer_with_geopandas

            return buffer_with_geopandas(points, distance)
        except ImportError:
            # Fallback: create square buffers
            logger.warning("geopandas not available, using simple square buffers")
            rings = []
            for coord in points.coordinates:
                x, y = coord[0], coord[1]
                ring = np.array(
                    [
                        [x - distance, y - distance],
                        [x + distance, y - distance],
                        [x + distance, y + distance],
                        [x - distance, y + distance],
                        [x - distance, y - distance],  # Close ring
                    ]
                )
                rings.append([ring])
            return PolygonSet(rings=rings, index=points.index)

    def _buffer_polygons(self, polygons: PolygonSet, distance: float) -> PolygonSet:
        """Buffer polygons."""
        try:
            from geosmith.tasks._geopandas_adapter import buffer_polygons_with_geopandas

            return buffer_polygons_with_geopandas(polygons, distance)
        except ImportError:
            # Fallback: expand bounding boxes
            logger.warning("geopandas not available, using simple bbox expansion")
            rings = []
            for polygon_rings in polygons.rings:
                if not polygon_rings:
                    continue
                exterior = polygon_rings[0].copy()
                # Simple expansion
                center = exterior.mean(axis=0)
                exterior = (exterior - center) * (1 + distance / 100) + center
                rings.append([exterior])
            return PolygonSet(rings=rings, index=polygons.index)

    def spatial_join(
        self,
        left: Union[GeoTable, PolygonSet, PointSet],
        right: Union[GeoTable, PolygonSet, PointSet],
        how: str = "inner",
        predicate: str = "intersects",
    ) -> pd.DataFrame:
        """Perform spatial join between two geometries.

        Args:
            left: Left geometry.
            right: Right geometry.
            how: Join type ('inner', 'left', 'right').
            predicate: Spatial predicate ('intersects', 'within', 'contains').

        Returns:
            DataFrame with join results.
        """
        logger.info(f"Performing spatial join: {how}, {predicate}")

        # Convert to objects
        left_obj = self._to_object(left)
        right_obj = self._to_object(right)

        # Perform spatial join
        if predicate == "intersects":
            if isinstance(left_obj, PointSet) and isinstance(right_obj, PolygonSet):
                mask = point_in_polygon(left_obj, right_obj)
                # Create result DataFrame
                left_indices, right_indices = np.where(mask)
                result = pd.DataFrame(
                    {
                        "left_idx": left_indices,
                        "right_idx": right_indices,
                    }
                )
                return result
            else:
                raise ValueError(
                    f"Unsupported join combination: {type(left_obj)}, {type(right_obj)}"
                )
        else:
            raise ValueError(f"Unsupported predicate: {predicate}")

    def distance_to_nearest(
        self,
        query: Union[GeoTable, PointSet],
        target: Union[GeoTable, PointSet],
        k: int = 1,
    ) -> pd.DataFrame:
        """Compute distance to nearest k neighbors.

        Args:
            query: Query geometry.
            target: Target geometry.
            k: Number of neighbors.

        Returns:
            DataFrame with distances and indices.
        """
        logger.info(f"Computing distance to {k} nearest neighbors")

        query_obj = self._to_pointset(query)
        target_obj = self._to_pointset(target)

        indices, distances = nearest_neighbor_search(query_obj, target_obj, k=k)

        # Create result DataFrame
        result_data = {}
        for i in range(k):
            result_data[f"neighbor_{i}_idx"] = indices[:, i]
            result_data[f"neighbor_{i}_dist"] = distances[:, i]

        return pd.DataFrame(result_data)

    def _to_object(
        self, geometry: Union[GeoTable, PolygonSet, PointSet]
    ) -> Union[PolygonSet, PointSet]:
        """Convert input to Layer 1 object."""
        if isinstance(geometry, GeoTable):
            geoms = geometry.data[geometry.geometry_column].tolist()
            if isinstance(geoms[0], PointSet):
                # Combine all points
                all_coords = np.vstack([g.coordinates for g in geoms])
                return PointSet(coordinates=all_coords, index=geometry.index)
            elif isinstance(geoms[0], PolygonSet):
                # Combine all polygons
                all_rings = []
                for poly in geoms:
                    all_rings.extend(poly.rings)
                return PolygonSet(rings=all_rings, index=geometry.index)
            else:
                raise ValueError(f"Unsupported geometry type: {type(geoms[0])}")
        elif isinstance(geometry, (PointSet, PolygonSet)):
            return geometry
        else:
            raise ValueError(f"Unsupported input type: {type(geometry)}")

    def _to_pointset(self, geometry: Union[GeoTable, PointSet]) -> PointSet:
        """Convert input to PointSet."""
        if isinstance(geometry, GeoTable):
            geoms = geometry.data[geometry.geometry_column].tolist()
            if isinstance(geoms[0], PointSet):
                all_coords = np.vstack([g.coordinates for g in geoms])
                return PointSet(coordinates=all_coords, index=geometry.index)
            else:
                raise ValueError("Geometry must be PointSet for distance operations")
        elif isinstance(geometry, PointSet):
            return geometry
        else:
            raise ValueError(
                f"Input must be PointSet or GeoTable with PointSet, "
                f"got {type(geometry)}"
            )
