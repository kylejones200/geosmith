"""Tests for Layer 2 Primitives."""

import numpy as np
import pytest

from geosmith.objects import PointSet, PolygonSet
from geosmith.primitives.geometry import (
    bounding_box,
    line_length,
    nearest_neighbor_search,
    polygon_area,
)
from geosmith.primitives.raster import grid_resample, zonal_reduce
from geosmith.objects import LineSet, RasterGrid


class TestGeometryPrimitives:
    """Tests for geometry primitives."""

    def test_nearest_neighbor_search(self):
        """Test nearest neighbor search."""
        query_coords = np.array([[0, 0], [5, 5]])
        target_coords = np.array([[1, 1], [2, 2], [3, 3]])
        query = PointSet(coordinates=query_coords)
        target = PointSet(coordinates=target_coords)

        indices, distances = nearest_neighbor_search(query, target, k=1)
        assert indices.shape == (2, 1)
        assert distances.shape == (2, 1)

    def test_bounding_box(self):
        """Test bounding box computation."""
        coords = np.array([[0, 0], [1, 1], [2, 2]])
        points = PointSet(coordinates=coords)
        bbox = bounding_box(points=points)
        assert bbox == (0.0, 0.0, 2.0, 2.0)

    def test_polygon_area(self):
        """Test polygon area computation."""
        ring = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
        polygons = PolygonSet(rings=[[ring]])
        area = polygon_area(polygons, poly_idx=0)
        assert abs(area - 1.0) < 0.01  # Square of side 1

    def test_line_length(self):
        """Test line length computation."""
        vertices = [np.array([[0, 0], [1, 0], [1, 1]])]
        line = LineSet(vertices=vertices)
        length = line_length(line, line_idx=0)
        assert abs(length - 2.0) < 0.01  # Two segments of length 1


class TestRasterPrimitives:
    """Tests for raster primitives."""

    def test_grid_resample(self):
        """Test grid resampling."""
        data = np.random.rand(10, 10)
        transform = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
        source = RasterGrid(data=data, transform=transform)

        target_transform = (2.0, 0.0, 0.0, 0.0, -2.0, 0.0)
        target_shape = (5, 5)
        result = grid_resample(source, target_transform, target_shape)

        assert result.data.shape == (5, 5)

    def test_zonal_reduce(self):
        """Test zonal reduction."""
        data = np.random.rand(10, 10) * 100
        transform = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
        raster = RasterGrid(data=data, transform=transform)

        # Create simple polygon
        ring = np.array([[2, 2], [8, 2], [8, 8], [2, 8], [2, 2]])
        polygons = PolygonSet(rings=[[ring]])

        stats = zonal_reduce(raster, polygons, reducer="mean")
        assert len(stats) == 1
        assert 0 <= stats[0] <= 100

