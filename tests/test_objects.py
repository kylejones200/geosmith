"""Tests for Layer 1 Objects."""

import numpy as np
import pandas as pd
import pytest

from geosmith.objects import (
    GeoIndex,
    GeoTable,
    LineSet,
    PointSet,
    PolygonSet,
    RasterGrid,
)


class TestGeoIndex:
    """Tests for GeoIndex."""

    def test_valid_geoindex(self):
        """Test creating valid GeoIndex."""
        idx = GeoIndex(crs="EPSG:4326", bounds=(0, 0, 10, 10))
        assert idx.crs == "EPSG:4326"
        assert idx.bounds == (0, 0, 10, 10)
        assert idx.axis_order == "xy"

    def test_invalid_bounds(self):
        """Test invalid bounds raise ValueError."""
        with pytest.raises(ValueError, match="minx.*maxx"):
            GeoIndex(crs="EPSG:4326", bounds=(10, 0, 0, 10))

        with pytest.raises(ValueError, match="miny.*maxy"):
            GeoIndex(crs="EPSG:4326", bounds=(0, 10, 10, 0))

    def test_invalid_axis_order(self):
        """Test invalid axis_order raises ValueError."""
        with pytest.raises(ValueError, match="axis_order"):
            GeoIndex(crs="EPSG:4326", bounds=(0, 0, 10, 10), axis_order="invalid")


class TestPointSet:
    """Tests for PointSet."""

    def test_valid_pointset(self):
        """Test creating valid PointSet."""
        coords = np.array([[0, 0], [1, 1], [2, 2]])
        points = PointSet(coordinates=coords)
        assert len(points.coordinates) == 3
        assert points.attributes is None
        assert points.index is None

    def test_pointset_with_attributes(self):
        """Test PointSet with attributes."""
        coords = np.array([[0, 0], [1, 1]])
        attrs = pd.DataFrame({"value": [1, 2]})
        points = PointSet(coordinates=coords, attributes=attrs)
        assert len(points.attributes) == 2

    def test_invalid_coordinates(self):
        """Test invalid coordinates raise ValueError."""
        with pytest.raises(ValueError, match="coordinates must be 2D"):
            PointSet(coordinates=np.array([0, 0]))

        with pytest.raises(ValueError, match="coordinates must have 2 or 3 dimensions"):
            PointSet(coordinates=np.array([[0]]))

    def test_mismatched_attributes(self):
        """Test mismatched attributes raise ValueError."""
        coords = np.array([[0, 0], [1, 1]])
        attrs = pd.DataFrame({"value": [1]})  # Wrong length
        with pytest.raises(ValueError, match="attributes length"):
            PointSet(coordinates=coords, attributes=attrs)


class TestLineSet:
    """Tests for LineSet."""

    def test_valid_lineset(self):
        """Test creating valid LineSet."""
        vertices = [
            np.array([[0, 0], [1, 1]]),
            np.array([[2, 2], [3, 3]]),
        ]
        lines = LineSet(vertices=vertices)
        assert len(lines.vertices) == 2

    def test_invalid_line(self):
        """Test invalid line raises ValueError."""
        with pytest.raises(ValueError, match="must have at least 2 vertices"):
            LineSet(vertices=[np.array([[0, 0]])])


class TestPolygonSet:
    """Tests for PolygonSet."""

    def test_valid_polygonset(self):
        """Test creating valid PolygonSet."""
        rings = [
            [np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])],
        ]
        polygons = PolygonSet(rings=rings)
        assert len(polygons.rings) == 1

    def test_invalid_polygon(self):
        """Test invalid polygon raises ValueError."""
        with pytest.raises(ValueError, match="must have at least 3 vertices"):
            PolygonSet(rings=[[np.array([[0, 0], [1, 1]])]])


class TestRasterGrid:
    """Tests for RasterGrid."""

    def test_valid_rastergrid_2d(self):
        """Test creating valid 2D RasterGrid."""
        data = np.random.rand(10, 10)
        transform = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
        raster = RasterGrid(data=data, transform=transform)
        assert raster.data.shape == (10, 10)

    def test_valid_rastergrid_3d(self):
        """Test creating valid 3D RasterGrid."""
        data = np.random.rand(3, 10, 10)
        transform = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
        band_names = ["red", "green", "blue"]
        raster = RasterGrid(data=data, transform=transform, band_names=band_names)
        assert raster.data.shape == (3, 10, 10)

    def test_mismatched_band_names(self):
        """Test mismatched band names raise ValueError."""
        data = np.random.rand(3, 10, 10)
        transform = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
        band_names = ["red", "green"]  # Wrong length
        with pytest.raises(ValueError, match="band_names length"):
            RasterGrid(data=data, transform=transform, band_names=band_names)


class TestGeoTable:
    """Tests for GeoTable."""

    def test_valid_geotable(self):
        """Test creating valid GeoTable."""
        coords1 = np.array([[0, 0]])
        coords2 = np.array([[1, 1]])
        points1 = PointSet(coordinates=coords1)
        points2 = PointSet(coordinates=coords2)
        data = pd.DataFrame({"id": [1, 2], "geometry": [points1, points2]})
        table = GeoTable(data=data, geometry_column="geometry")
        assert len(table.data) == 2

    def test_missing_geometry_column(self):
        """Test missing geometry column raises ValueError."""
        data = pd.DataFrame({"id": [1, 2]})
        with pytest.raises(ValueError, match="geometry column"):
            GeoTable(data=data, geometry_column="geometry")

