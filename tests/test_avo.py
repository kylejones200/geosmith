"""Tests for AVO (Amplitude Versus Offset) calculations.

Migrated from geosuite.petro.avo.
"""

import numpy as np
import pytest

from geosmith.primitives.petrophysics import (
    calculate_avo_attributes,
    calculate_avo_from_slowness,
    calculate_velocities_from_slowness,
    preprocess_avo_inputs,
)


class TestCalculateVelocitiesFromSlowness:
    """Tests for calculate_velocities_from_slowness."""

    def test_m_s_units(self):
        """Test velocity calculation with m/s units."""
        dtc = np.array([100, 120, 140])  # μs/m
        dts = np.array([180, 200, 220])  # μs/m

        vp, vs = calculate_velocities_from_slowness(dtc, dts, units="m/s")

        # v = 1e6 / dt
        expected_vp = 1e6 / dtc
        expected_vs = 1e6 / dts

        np.testing.assert_allclose(vp, expected_vp, rtol=1e-10)
        np.testing.assert_allclose(vs, expected_vs, rtol=1e-10)

    def test_ft_s_units(self):
        """Test velocity calculation with ft/s units."""
        dtc = np.array([100, 120, 140])  # μs/ft
        dts = np.array([180, 200, 220])  # μs/ft

        vp, vs = calculate_velocities_from_slowness(dtc, dts, units="ft/s")

        # v = 1e6 / dt (in ft/s), then convert to m/s
        expected_vp_ft_s = 1e6 / dtc
        expected_vs_ft_s = 1e6 / dts
        expected_vp = expected_vp_ft_s / 3.281
        expected_vs = expected_vs_ft_s / 3.281

        np.testing.assert_allclose(vp, expected_vp, rtol=1e-10)
        np.testing.assert_allclose(vs, expected_vs, rtol=1e-10)

    def test_invalid_units(self):
        """Test that invalid units raise error."""
        dtc = np.array([100, 120, 140])
        dts = np.array([180, 200, 220])

        with pytest.raises(ValueError, match="Unknown units"):
            calculate_velocities_from_slowness(dtc, dts, units="invalid")

    def test_zero_slowness(self):
        """Test that zero slowness returns NaN."""
        dtc = np.array([100, 0, 140])
        dts = np.array([180, 200, 220])

        vp, vs = calculate_velocities_from_slowness(dtc, dts, units="m/s")

        assert np.isnan(vp[1])
        assert not np.isnan(vp[0])
        assert not np.isnan(vp[2])


class TestPreprocessAvoInputs:
    """Tests for preprocess_avo_inputs."""

    def test_basic_preprocessing(self):
        """Test basic preprocessing of AVO inputs."""
        vp = np.array([3000, 3100, 3200])
        vs = np.array([1500, 1550, 1600])
        rho = np.array([2.3, 2.4, 2.5])

        preprocessed = preprocess_avo_inputs(vp, vs, rho)

        assert "VP_AVG" in preprocessed
        assert "VS_AVG" in preprocessed
        assert "RHO" in preprocessed
        assert "dVp" in preprocessed
        assert "dVs" in preprocessed
        assert "dRho" in preprocessed

        # Check averages
        assert preprocessed["VP_AVG"][0] == (vp[0] + vp[1]) / 2.0
        assert preprocessed["VS_AVG"][0] == (vs[0] + vs[1]) / 2.0
        assert preprocessed["RHO"][0] == (rho[0] + rho[1]) / 2.0

        # Check differences
        assert preprocessed["dVp"][0] == vp[1] - vp[0]
        assert preprocessed["dVs"][0] == vs[1] - vs[0]
        assert preprocessed["dRho"][0] == rho[1] - rho[0]

    def test_mismatched_lengths(self):
        """Test that mismatched lengths raise error."""
        vp = np.array([3000, 3100, 3200])
        vs = np.array([1500, 1550])
        rho = np.array([2.3, 2.4, 2.5])

        with pytest.raises(ValueError, match="same length"):
            preprocess_avo_inputs(vp, vs, rho)

    def test_too_short(self):
        """Test that arrays with < 2 samples raise error."""
        vp = np.array([3000])
        vs = np.array([1500])
        rho = np.array([2.3])

        with pytest.raises(ValueError, match="at least 2 samples"):
            preprocess_avo_inputs(vp, vs, rho)


class TestCalculateAvoAttributes:
    """Tests for calculate_avo_attributes."""

    def test_basic_avo_calculation(self):
        """Test basic AVO attribute calculation."""
        vp = np.array([3000, 3100, 3200])
        vs = np.array([1500, 1550, 1600])
        rho = np.array([2.3, 2.4, 2.5])

        avo_dict = calculate_avo_attributes(vp, vs, rho, return_all=True)

        assert "A" in avo_dict  # Intercept
        assert "B" in avo_dict  # Gradient
        assert "PR" in avo_dict  # Poisson's Ratio
        assert "Rp" in avo_dict  # P-wave reflectivity
        assert "Rs" in avo_dict  # S-wave reflectivity
        assert "FF" in avo_dict  # Fluid Factor

        # Check that all arrays have correct length
        assert len(avo_dict["A"]) == len(vp)
        assert len(avo_dict["B"]) == len(vp)
        assert len(avo_dict["PR"]) == len(vp)

    def test_key_attributes_only(self):
        """Test returning only key attributes."""
        vp = np.array([3000, 3100, 3200])
        vs = np.array([1500, 1550, 1600])
        rho = np.array([2.3, 2.4, 2.5])

        avo_dict = calculate_avo_attributes(vp, vs, rho, return_all=False)

        assert "A" in avo_dict
        assert "B" in avo_dict
        assert "PR" in avo_dict
        assert "k" not in avo_dict  # Should not be in key attributes
        assert "C" not in avo_dict  # Should not be in key attributes

    def test_realistic_values(self):
        """Test with realistic velocity and density values."""
        vp = np.array([3500, 3600, 3700])  # m/s
        vs = np.array([1800, 1850, 1900])  # m/s
        rho = np.array([2.4, 2.5, 2.6])  # g/cc

        avo_dict = calculate_avo_attributes(vp, vs, rho)

        # Check that values are reasonable
        assert np.all(np.isfinite(avo_dict["A"]))
        assert np.all(np.isfinite(avo_dict["B"]))
        # Poisson's ratio should be between 0 and 0.5 for most rocks
        assert np.all((avo_dict["PR"] >= 0) | np.isnan(avo_dict["PR"]))
        assert np.all((avo_dict["PR"] <= 0.5) | np.isnan(avo_dict["PR"]))


class TestCalculateAvoFromSlowness:
    """Tests for calculate_avo_from_slowness."""

    def test_avo_from_slowness(self):
        """Test AVO calculation from slowness."""
        dtc = np.array([100, 120, 140])  # μs/ft
        dts = np.array([180, 200, 220])  # μs/ft
        rho = np.array([2.3, 2.4, 2.5])  # g/cc

        avo_dict = calculate_avo_from_slowness(dtc, dts, rho, units="ft/s")

        assert "A" in avo_dict
        assert "B" in avo_dict
        assert "PR" in avo_dict
        assert len(avo_dict["A"]) == len(dtc)


