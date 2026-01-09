"""Tests for stress inversion from wellbore failure observations.

Migrated from geosuite.geomech.stress_inversion.
"""

import numpy as np
import pytest

from geosmith.primitives.geomechanics import (
    invert_stress_combined,
    invert_stress_from_breakout,
    invert_stress_from_dif,
)

# Optional scipy dependency for optimization method
try:
    from scipy.optimize import minimize

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class TestInvertStressFromBreakout:
    """Tests for invert_stress_from_breakout."""

    def test_analytical_method_scalar(self):
        """Test analytical stress inversion with scalar inputs."""
        result = invert_stress_from_breakout(
            breakout_width=30.0,
            breakout_azimuth=45.0,
            depth=2000.0,
            sv=50.0,
            pp=20.0,
            method="analytical",
        )

        assert "shmax" in result
        assert "shmin" in result
        assert "stress_ratio" in result

        assert isinstance(result["shmax"], (float, np.ndarray))
        assert isinstance(result["shmin"], (float, np.ndarray))

        # Check physical constraints
        assert result["shmax"] > result["shmin"]
        assert result["shmin"] >= 20.0  # Should be >= pore pressure
        assert result["shmax"] <= 75.0  # Should be <= 1.5 * Sv

    def test_analytical_method_array(self):
        """Test analytical stress inversion with array inputs."""
        breakout_width = np.array([25.0, 30.0, 35.0])
        breakout_azimuth = np.array([45.0, 90.0, 135.0])
        depth = np.array([2000.0, 2100.0, 2200.0])
        sv = np.array([50.0, 52.5, 55.0])
        pp = np.array([20.0, 21.0, 22.0])

        result = invert_stress_from_breakout(
            breakout_width=breakout_width,
            breakout_azimuth=breakout_azimuth,
            depth=depth,
            sv=sv,
            pp=pp,
            method="analytical",
        )

        assert len(result["shmax"]) == len(breakout_width)
        assert len(result["shmin"]) == len(breakout_width)
        assert np.all(result["shmax"] > result["shmin"])

    @pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy not available")
    def test_optimization_method(self):
        """Test optimization-based stress inversion."""
        result = invert_stress_from_breakout(
            breakout_width=30.0,
            breakout_azimuth=45.0,
            depth=2000.0,
            sv=50.0,
            pp=20.0,
            method="optimization",
        )

        assert "shmax" in result
        assert "shmin" in result
        assert result["shmax"] > result["shmin"]

    def test_optimization_method_requires_scipy(self):
        """Test that optimization method raises ImportError without scipy."""
        if SCIPY_AVAILABLE:
            pytest.skip("scipy is available")

        with pytest.raises(ImportError, match="scipy is required"):
            invert_stress_from_breakout(
                breakout_width=30.0,
                breakout_azimuth=45.0,
                depth=2000.0,
                sv=50.0,
                pp=20.0,
                method="optimization",
            )


class TestInvertStressFromDif:
    """Tests for invert_stress_from_dif."""

    def test_basic_inversion(self):
        """Test basic DIF stress inversion."""
        result = invert_stress_from_dif(
            dif_azimuth=135.0, depth=2000.0, sv=50.0, pp=20.0
        )

        assert "shmax" in result
        assert "shmin" in result
        assert "stress_ratio" in result
        assert "shmin_azimuth" in result

        assert isinstance(result["shmax"], (float, np.ndarray))
        assert result["shmin_azimuth"] == pytest.approx(225.0, abs=1.0)  # 135 + 90

    def test_array_input(self):
        """Test DIF stress inversion with array inputs."""
        dif_azimuth = np.array([135.0, 180.0, 225.0])
        depth = np.array([2000.0, 2100.0, 2200.0])
        sv = np.array([50.0, 52.5, 55.0])
        pp = np.array([20.0, 21.0, 22.0])

        result = invert_stress_from_dif(
            dif_azimuth=dif_azimuth, depth=depth, sv=sv, pp=pp
        )

        assert len(result["shmax"]) == len(dif_azimuth)
        assert len(result["shmin_azimuth"]) == len(dif_azimuth)


class TestInvertStressCombined:
    """Tests for invert_stress_combined."""

    def test_breakout_only(self):
        """Test combined inversion with breakout data only."""
        result = invert_stress_combined(
            depth=2000.0,
            sv=50.0,
            pp=20.0,
            breakout_data={"width": np.array([30.0]), "azimuth": np.array([45.0])},
            dif_data=None,
        )

        assert "shmax" in result
        assert "shmin" in result
        assert "confidence" in result
        assert result["confidence"] == "medium"

    def test_dif_only(self):
        """Test combined inversion with DIF data only."""
        result = invert_stress_combined(
            depth=2000.0,
            sv=50.0,
            pp=20.0,
            breakout_data=None,
            dif_data={"azimuth": np.array([135.0])},
        )

        assert "shmax" in result
        assert "shmin" in result
        assert result["confidence"] == "low"

    def test_combined_data(self):
        """Test combined inversion with both breakout and DIF data."""
        result = invert_stress_combined(
            depth=np.array([2000.0]),
            sv=np.array([50.0]),
            pp=np.array([20.0]),
            breakout_data={
                "width": np.array([30.0]),
                "azimuth": np.array([45.0]),
            },
            dif_data={"azimuth": np.array([135.0])},
        )

        assert "shmax" in result
        assert "shmin" in result
        assert result["confidence"] == "high"

    def test_no_data_error(self):
        """Test that missing data raises error."""
        with pytest.raises(ValueError, match="At least one of"):
            invert_stress_combined(
                depth=2000.0, sv=50.0, pp=20.0, breakout_data=None, dif_data=None
            )

