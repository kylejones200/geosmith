"""Tests for petrophysics primitives."""

import numpy as np
import pytest

from geosmith.primitives.petrophysics import (
    ArchieParams,
    calculate_bulk_volume_water,
    calculate_water_saturation,
    pickett_isolines,
)


class TestArchieParams:
    """Tests for ArchieParams."""

    def test_valid_params(self):
        """Test creating valid ArchieParams."""
        params = ArchieParams(a=1.0, m=2.0, n=2.0, rw=0.1)
        assert params.a == 1.0
        assert params.m == 2.0

    def test_invalid_params(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError, match="must be > 0"):
            ArchieParams(a=-1.0)

        with pytest.raises(ValueError, match="must be > 0"):
            ArchieParams(m=0.0)


class TestCalculateWaterSaturation:
    """Tests for calculate_water_saturation."""

    def test_water_saturation(self):
        """Test water saturation calculation."""
        params = ArchieParams(a=1.0, m=2.0, n=2.0, rw=0.1)

        # Simple case: rt=10, phi=0.25
        # Sw^2 = (1.0 * 0.1) / (10 * 0.25^2) = 0.1 / 0.625 = 0.16
        # Sw = sqrt(0.16) = 0.4
        sw = calculate_water_saturation(rt=10.0, phi=0.25, params=params)

        assert sw > 0
        assert sw <= 1.0
        assert np.isfinite(sw)

    def test_water_saturation_array(self):
        """Test water saturation with arrays."""
        params = ArchieParams()
        rt = np.array([10.0, 20.0, 30.0])
        phi = np.array([0.25, 0.30, 0.35])

        sw = calculate_water_saturation(rt, phi, params)

        assert len(sw) == 3
        assert np.all(sw > 0)
        assert np.all(sw <= 1.0)


class TestCalculateBulkVolumeWater:
    """Tests for calculate_bulk_volume_water."""

    def test_bvw(self):
        """Test bulk volume water calculation."""
        bvw = calculate_bulk_volume_water(phi=0.25, sw=0.5)
        assert bvw == pytest.approx(0.125)

    def test_bvw_array(self):
        """Test BVW with arrays."""
        phi = np.array([0.25, 0.30, 0.35])
        sw = np.array([0.5, 0.6, 0.7])

        bvw = calculate_bulk_volume_water(phi, sw)
        assert len(bvw) == 3
        assert bvw[0] == pytest.approx(0.125)


class TestPickettIsolines:
    """Tests for pickett_isolines."""

    def test_pickett_isolines(self):
        """Test Pickett isoline generation."""
        params = ArchieParams()
        phi_vals = np.array([0.1, 0.3])
        sw_vals = np.array([0.5, 0.7, 1.0])

        isolines = pickett_isolines(phi_vals, sw_vals, params)

        assert len(isolines) == 3
        for phi, rt, label in isolines:
            assert len(phi) == 100  # num_points default
            assert len(rt) == 100
            assert "Sw=" in label

