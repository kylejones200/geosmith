"""Tests for geomechanics primitives."""

import numpy as np
import pytest

from geosmith.primitives.geomechanics import (
    calculate_effective_stress,
    calculate_overpressure,
    calculate_pressure_gradient,
    calculate_stress_ratio,
    pressure_to_mud_weight,
)


class TestCalculateEffectiveStress:
    """Tests for calculate_effective_stress."""

    def test_effective_stress(self):
        """Test effective stress calculation."""
        sv_eff = calculate_effective_stress(sv=50.0, pp=20.0, biot=1.0)
        assert sv_eff == pytest.approx(30.0)

    def test_effective_stress_with_biot(self):
        """Test effective stress with Biot coefficient."""
        sv_eff = calculate_effective_stress(sv=50.0, pp=20.0, biot=0.8)
        assert sv_eff == pytest.approx(34.0)  # 50 - 0.8*20


class TestCalculateOverpressure:
    """Tests for calculate_overpressure."""

    def test_overpressure(self):
        """Test overpressure calculation."""
        overpressure = calculate_overpressure(pp=25.0, ph=20.0)
        assert overpressure == pytest.approx(5.0)


class TestCalculatePressureGradient:
    """Tests for calculate_pressure_gradient."""

    def test_pressure_gradient(self):
        """Test pressure gradient calculation."""
        pressure = np.array([10.0, 20.0, 30.0])
        depth = np.array([1000.0, 2000.0, 3000.0])

        gradient = calculate_pressure_gradient(pressure, depth)

        assert len(gradient) == 3
        # Gradient should be ~0.01 MPa/m (10 MPa per 1000 m)
        assert gradient[1] == pytest.approx(0.01, abs=0.001)


class TestPressureToMudWeight:
    """Tests for pressure_to_mud_weight."""

    def test_mud_weight(self):
        """Test mud weight conversion."""
        mw = pressure_to_mud_weight(pressure=20.0, depth=2000.0)

        # MW = (20e6 Pa) / (9.81 * 2000) / 1000 = ~1.02 g/cc
        assert mw > 1.0
        assert mw < 2.0


class TestCalculateStressRatio:
    """Tests for calculate_stress_ratio."""

    def test_stress_ratio(self):
        """Test stress ratio calculation."""
        k = calculate_stress_ratio(shmin=40.0, sv=50.0)
        assert k == pytest.approx(0.8)

