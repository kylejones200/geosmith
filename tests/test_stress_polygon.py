"""Tests for stress polygon limits.

Migrated from geosuite.geomech.stress_polygon.
"""

import numpy as np
import pytest

from geosmith.primitives.geomechanics import stress_polygon_limits


class TestStressPolygonLimits:
    """Tests for stress_polygon_limits."""

    def test_scalar_input(self):
        """Test stress polygon limits with scalar inputs."""
        limits = stress_polygon_limits(sv=50.0, pp=20.0, shmin=30.0)

        assert "normal" in limits
        assert "strike_slip" in limits
        assert "reverse" in limits

        nf_min, nf_max = limits["normal"]
        ss_min, ss_max = limits["strike_slip"]
        rf_min, rf_max = limits["reverse"]

        assert isinstance(nf_min, float)
        assert isinstance(nf_max, float)
        assert isinstance(ss_min, float)
        assert isinstance(ss_max, float)
        assert isinstance(rf_min, float)
        assert isinstance(rf_max, float)

        # Normal faulting: Sv > SHmax > Shmin
        assert nf_max == 50.0  # Should equal Sv
        assert nf_min < nf_max

        # Strike-slip: SHmax > Sv > Shmin
        assert ss_min == 50.0  # Should equal Sv
        assert ss_max > ss_min

        # Reverse: SHmax > Shmin > Sv
        assert rf_min > ss_max  # Should be above strike-slip
        assert rf_max > rf_min

    def test_array_input(self):
        """Test stress polygon limits with array inputs."""
        sv = np.array([50.0, 60.0, 70.0])
        pp = np.array([20.0, 25.0, 30.0])
        shmin = np.array([30.0, 35.0, 40.0])

        limits = stress_polygon_limits(sv=sv, pp=pp, shmin=shmin)

        nf_min, nf_max = limits["normal"]
        ss_min, ss_max = limits["strike_slip"]
        rf_min, rf_max = limits["reverse"]

        assert isinstance(nf_min, np.ndarray)
        assert isinstance(nf_max, np.ndarray)
        assert len(nf_min) == 3
        assert len(nf_max) == 3

        # Check all values are reasonable
        assert np.all(nf_max == sv)
        assert np.all(nf_min < nf_max)
        assert np.all(ss_min == sv)
        assert np.all(ss_max > ss_min)

    def test_without_shmin(self):
        """Test stress polygon limits without shmin."""
        limits = stress_polygon_limits(sv=50.0, pp=20.0, shmin=None)

        rf_min, rf_max = limits["reverse"]

        assert rf_max is None
        assert rf_min > 0

    def test_different_mu(self):
        """Test stress polygon limits with different friction coefficient."""
        limits1 = stress_polygon_limits(sv=50.0, pp=20.0, mu=0.6)
        limits2 = stress_polygon_limits(sv=50.0, pp=20.0, mu=1.0)

        # Higher mu should give wider range for strike-slip
        ss_min1, ss_max1 = limits1["strike_slip"]
        ss_min2, ss_max2 = limits2["strike_slip"]

        assert ss_max2 > ss_max1

    def test_mismatched_lengths(self):
        """Test that mismatched array lengths raise error."""
        sv = np.array([50.0, 60.0])
        pp = np.array([20.0])

        with pytest.raises(ValueError, match="must have same length"):
            stress_polygon_limits(sv=sv, pp=pp)

    def test_realistic_values(self):
        """Test with realistic geomechanical values."""
        # Typical Gulf of Mexico values
        sv = 50.0  # MPa at ~2000m
        pp = 20.0  # MPa (normal pressure)
        shmin = 30.0  # MPa

        limits = stress_polygon_limits(sv=sv, pp=pp, shmin=shmin, mu=0.6)

        nf_min, nf_max = limits["normal"]
        ss_min, ss_max = limits["strike_slip"]
        rf_min, rf_max = limits["reverse"]

        # Normal faulting should allow lower SHmax
        assert nf_max == sv
        assert nf_min < sv

        # Strike-slip should allow higher SHmax
        assert ss_min == sv
        assert ss_max > sv

        # Reverse should allow highest SHmax
        assert rf_min > ss_max
        assert rf_max > rf_min

