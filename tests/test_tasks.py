"""Tests for Layer 3 Tasks."""

import numpy as np

from geosmith.objects import PointSet, PolygonSet
from geosmith.tasks.featuretask import FeatureTask
from geosmith.tasks.rastertask import RasterTask


class TestFeatureTask:
    """Tests for FeatureTask."""

    def test_buffer_points(self):
        """Test buffering points."""
        coords = np.array([[0, 0], [1, 1]])
        points = PointSet(coordinates=coords)
        task = FeatureTask()

        buffered = task.buffer(points, distance=0.5)
        assert isinstance(buffered, PolygonSet)
        assert len(buffered.rings) == 2

    def test_distance_to_nearest(self):
        """Test distance to nearest neighbor."""
        query_coords = np.array([[0, 0]])
        target_coords = np.array([[1, 1], [2, 2]])
        query = PointSet(coordinates=query_coords)
        target = PointSet(coordinates=target_coords)

        task = FeatureTask()
        result = task.distance_to_nearest(query, target, k=1)
        assert len(result) == 1
        assert "neighbor_0_idx" in result.columns
        assert "neighbor_0_dist" in result.columns


class TestRasterTask:
    """Tests for RasterTask."""

    def test_resample(self):
        """Test raster resampling."""
        from geosmith.objects import RasterGrid

        data = np.random.rand(10, 10)
        transform = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
        raster = RasterGrid(data=data, transform=transform)

        task = RasterTask()
        target_transform = (2.0, 0.0, 0.0, 0.0, -2.0, 0.0)
        result = task.resample(raster, target_transform, (5, 5))

        assert result.data.shape == (5, 5)


