"""Tests for fracture orientation and stress inversion.

Migrated from geosuite.geomech.fracture_orientation and geosuite.geomech.stress_inversion.
"""

import numpy as np
import pytest

from geosmith.primitives.geomechanics import (
    calculate_fracture_aperture,
    calculate_fracture_permeability,
    predict_fracture_orientation,
)

# Optional scipy dependency for fracture_orientation_distribution
try:
    from scipy.stats import vonmises  # noqa: F401

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Import fracture_orientation_distribution conditionally
if SCIPY_AVAILABLE:
    from geosmith.primitives.geomechanics import fracture_orientation_distribution


class TestPredictFractureOrientation:
    """Tests for predict_fracture_orientation."""

    def test_coulomb_method_scalar(self):
        """Test Coulomb fracture prediction with scalar inputs."""
        result = predict_fracture_orientation(
            shmax_azimuth=45.0, shmin_azimuth=135.0, stress_ratio=1.2, method="coulomb"
        )

        assert "strike" in result
        assert "dip" in result
        assert "type" in result
        assert "azimuth" in result

        assert isinstance(result["strike"], (float, np.ndarray))
        assert isinstance(result["dip"], (float, np.ndarray))

    def test_coulomb_method_array(self):
        """Test Coulomb fracture prediction with array inputs."""
        shmax_az = np.array([45.0, 90.0, 135.0])
        shmin_az = np.array([135.0, 180.0, 225.0])
        stress_ratio = np.array([0.8, 1.2, 1.5])

        result = predict_fracture_orientation(
            shmax_azimuth=shmax_az,
            shmin_azimuth=shmin_az,
            stress_ratio=stress_ratio,
            method="coulomb",
        )

        assert len(result["strike"]) == len(shmax_az)
        assert len(result["dip"]) == len(shmax_az)
        assert len(result["type"]) == len(shmax_az)

    def test_griffith_method(self):
        """Test Griffith fracture prediction."""
        result = predict_fracture_orientation(
            shmax_azimuth=45.0, shmin_azimuth=135.0, stress_ratio=1.2, method="griffith"
        )

        assert result["dip"] == pytest.approx(90.0)  # Should be vertical
        assert result["type"] == "tensile"

    def test_tensile_method(self):
        """Test tensile fracture prediction."""
        result = predict_fracture_orientation(
            shmax_azimuth=45.0, shmin_azimuth=135.0, stress_ratio=1.2, method="tensile"
        )

        assert result["dip"] == pytest.approx(90.0)  # Should be vertical
        assert result["type"] == "tensile"

    def test_invalid_method(self):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Unknown method"):
            predict_fracture_orientation(
                shmax_azimuth=45.0, shmin_azimuth=135.0, stress_ratio=1.2, method="invalid"
            )


class TestFractureOrientationDistribution:
    """Tests for fracture_orientation_distribution."""

    @pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy not available")
    def test_basic_distribution(self):
        """Test basic fracture orientation distribution."""
        strikes = fracture_orientation_distribution(
            mean_strike=45.0, concentration=10.0, n_samples=1000
        )

        assert isinstance(strikes, np.ndarray)
        assert len(strikes) == 1000
        assert np.all((strikes >= 0) & (strikes < 360))

        # Mean should be close to 45 degrees (accounting for circular nature)
        mean_strike = np.mean(strikes)
        assert mean_strike == pytest.approx(45.0, abs=30.0)  # Allow ±30° for circular

    @pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy not available")
    def test_high_concentration(self):
        """Test high concentration produces tight distribution."""
        strikes_high = fracture_orientation_distribution(
            mean_strike=45.0, concentration=50.0, n_samples=1000
        )
        strikes_low = fracture_orientation_distribution(
            mean_strike=45.0, concentration=1.0, n_samples=1000
        )

        # High concentration should have lower standard deviation
        std_high = np.std(strikes_high)
        std_low = np.std(strikes_low)

        # Account for circular statistics (this is approximate)
        assert std_high < std_low * 2  # High concentration should be tighter

    def test_scipy_unavailable(self):
        """Test that missing scipy raises ImportError."""
        if SCIPY_AVAILABLE:
            pytest.skip("scipy is available")

        with pytest.raises(ImportError, match="scipy is required"):
            fracture_orientation_distribution(mean_strike=45.0)


class TestCalculateFractureAperture:
    """Tests for calculate_fracture_aperture."""

    def test_scalar_input(self):
        """Test fracture aperture calculation with scalar input."""
        aperture = calculate_fracture_aperture(
            normal_stress=20.0, closure_stress=5.0, initial_aperture=0.1, stiffness=10.0
        )

        assert isinstance(aperture, float)
        assert aperture >= 0.0
        assert aperture <= 0.1  # Should not exceed initial aperture

    def test_array_input(self):
        """Test fracture aperture calculation with array input."""
        normal_stress = np.array([10.0, 20.0, 30.0])
        aperture = calculate_fracture_aperture(
            normal_stress=normal_stress,
            closure_stress=5.0,
            initial_aperture=0.1,
            stiffness=10.0,
        )

        assert isinstance(aperture, np.ndarray)
        assert len(aperture) == len(normal_stress)
        assert np.all(aperture >= 0.0)
        assert np.all(aperture <= 0.1)

    def test_closure_stress(self):
        """Test that fracture closes at closure stress."""
        aperture = calculate_fracture_aperture(
            normal_stress=5.0, closure_stress=5.0, initial_aperture=0.1, stiffness=10.0
        )

        assert aperture == pytest.approx(0.1, rel=1e-10)  # Should be at initial


class TestCalculateFracturePermeability:
    """Tests for calculate_fracture_permeability."""

    def test_scalar_input(self):
        """Test fracture permeability calculation with scalar input."""
        k = calculate_fracture_permeability(aperture=0.1, spacing=1.0)

        assert isinstance(k, float)
        assert k > 0

        # Check cubic law: k should be proportional to aperture^3
        k1 = calculate_fracture_permeability(aperture=0.1, spacing=1.0)
        k2 = calculate_fracture_permeability(aperture=0.2, spacing=1.0)

        # k2 should be approximately 8x k1 (2^3 = 8)
        assert k2 == pytest.approx(k1 * 8, rel=0.1)

    def test_array_input(self):
        """Test fracture permeability calculation with array input."""
        aperture = np.array([0.1, 0.2, 0.3])
        k = calculate_fracture_permeability(aperture=aperture, spacing=1.0)

        assert isinstance(k, np.ndarray)
        assert len(k) == len(aperture)
        assert np.all(k > 0)

    def test_spacing_effect(self):
        """Test that permeability decreases with spacing."""
        k1 = calculate_fracture_permeability(aperture=0.1, spacing=1.0)
        k2 = calculate_fracture_permeability(aperture=0.1, spacing=2.0)

        # k2 should be approximately half of k1
        assert k2 == pytest.approx(k1 / 2, rel=0.1)


