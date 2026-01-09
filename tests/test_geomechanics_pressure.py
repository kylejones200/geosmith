"""Tests for geomechanics pressure and stress calculations.

Migrated from geosuite.geomech.stresses and geosuite.geomech.pressures.
"""

import numpy as np
import pytest

from geosmith.primitives.geomechanics import (
    calculate_hydrostatic_pressure,
    calculate_overburden_stress,
    estimate_shmin_from_poisson,
)


class TestEstimateShminFromPoisson:
    """Tests for estimate_shmin_from_poisson."""

    def test_scalar_input(self):
        """Test Shmin estimation with scalar inputs."""
        sv = 50.0  # MPa
        pp = 20.0  # MPa
        nu = 0.25
        biot = 1.0

        shmin = estimate_shmin_from_poisson(sv, pp, nu, biot)

        # Shmin = (ν / (1 - ν)) * (Sv - α*Pp) + α*Pp
        expected = (nu / (1 - nu)) * (sv - biot * pp) + biot * pp
        assert shmin == pytest.approx(expected, rel=1e-10)

    def test_array_input(self):
        """Test Shmin estimation with array inputs."""
        sv = np.array([50.0, 60.0, 70.0])
        pp = np.array([20.0, 25.0, 30.0])
        nu = 0.25

        shmin = estimate_shmin_from_poisson(sv, pp, nu)

        assert isinstance(shmin, np.ndarray)
        assert len(shmin) == len(sv)

        # Check first value
        expected = (nu / (1 - nu)) * (sv[0] - pp[0]) + pp[0]
        assert shmin[0] == pytest.approx(expected, rel=1e-10)

    def test_mismatched_lengths(self):
        """Test that mismatched array lengths raise error."""
        sv = np.array([50.0, 60.0])
        pp = np.array([20.0])

        with pytest.raises(ValueError, match="same length"):
            estimate_shmin_from_poisson(sv, pp)

    def test_invalid_poissons_ratio(self):
        """Test that invalid Poisson's ratio raises error."""
        sv = 50.0
        pp = 20.0

        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            estimate_shmin_from_poisson(sv, pp, nu=1.0)

        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            estimate_shmin_from_poisson(sv, pp, nu=0.0)


class TestCalculateOverburdenStress:
    """Tests for calculate_overburden_stress."""

    def test_uniform_density(self):
        """Test overburden stress with uniform density."""
        depth = np.linspace(0, 3000, 1000)  # 0-3000m
        rhob = np.ones(1000) * 2.5  # 2.5 g/cc uniform
        g = 9.81

        sv = calculate_overburden_stress(depth, rhob, g)

        assert isinstance(sv, np.ndarray)
        assert len(sv) == len(depth)
        assert sv[0] == pytest.approx(0.0, abs=1e-10)  # At surface

        # Check that stress increases with depth
        assert np.all(np.diff(sv) >= 0)

        # Check final value: Sv = rho * g * depth / 1e6
        expected_final = 2.5 * 1000 * g * depth[-1] / 1e6
        assert sv[-1] == pytest.approx(expected_final, rel=1e-6)

    def test_variable_density(self):
        """Test overburden stress with variable density."""
        depth = np.array([0, 1000, 2000, 3000])
        rhob = np.array([2.0, 2.3, 2.5, 2.7])  # Increasing with depth
        g = 9.81

        sv = calculate_overburden_stress(depth, rhob, g)

        assert len(sv) == len(depth)
        assert sv[0] == pytest.approx(0.0, abs=1e-10)

        # Stress should increase with depth
        assert np.all(np.diff(sv) >= 0)

    def test_mismatched_lengths(self):
        """Test that mismatched lengths raise error."""
        depth = np.array([0, 1000, 2000])
        rhob = np.array([2.5, 2.5])

        with pytest.raises(ValueError, match="same length"):
            calculate_overburden_stress(depth, rhob)

    def test_empty_inputs(self):
        """Test that empty inputs raise error."""
        depth = np.array([])
        rhob = np.array([])

        with pytest.raises(ValueError, match="must not be empty"):
            calculate_overburden_stress(depth, rhob)


class TestCalculateHydrostaticPressure:
    """Tests for calculate_hydrostatic_pressure."""

    def test_scalar_input(self):
        """Test hydrostatic pressure with scalar depth."""
        depth = 2000.0  # meters
        rho_water = 1.03  # g/cc
        g = 9.81

        ph = calculate_hydrostatic_pressure(depth, rho_water, g)

        # Ph = rho_water * g * depth / 1e6
        expected = 1.03 * 1000 * g * depth / 1e6
        assert ph == pytest.approx(expected, rel=1e-10)

    def test_array_input(self):
        """Test hydrostatic pressure with array depth."""
        depth = np.array([1000, 2000, 3000])
        rho_water = 1.03
        g = 9.81

        ph = calculate_hydrostatic_pressure(depth, rho_water, g)

        assert isinstance(ph, np.ndarray)
        assert len(ph) == len(depth)

        # Check first value
        expected = 1.03 * 1000 * g * depth[0] / 1e6
        assert ph[0] == pytest.approx(expected, rel=1e-10)

        # Pressure should increase linearly with depth
        assert np.all(np.diff(ph) > 0)

    def test_empty_input(self):
        """Test that empty input raises error."""
        depth = np.array([])

        with pytest.raises(ValueError, match="must not be empty"):
            calculate_hydrostatic_pressure(depth)

    def test_realistic_values(self):
        """Test with realistic seawater values."""
        depth = np.array([1000, 2000, 3000])
        rho_water = 1.03  # Typical seawater density

        ph = calculate_hydrostatic_pressure(depth, rho_water)

        # At 2000m, pressure should be ~20 MPa
        assert ph[1] == pytest.approx(20.0, rel=0.1)

