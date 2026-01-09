"""Tests for failure criteria primitives."""

import numpy as np
import pytest

from geosmith.primitives.geomechanics import (
    drucker_prager_failure,
    hoek_brown_failure,
    mohr_coulomb_failure,
)


class TestMohrCoulombFailure:
    """Tests for mohr_coulomb_failure."""

    def test_mohr_coulomb_basic(self):
        """Test basic Mohr-Coulomb calculation."""
        sigma1 = 50.0
        sigma3 = 20.0
        sigma1_fail, safety = mohr_coulomb_failure(
            sigma1=sigma1, sigma3=sigma3, cohesion=10.0, friction_angle=30.0
        )

        assert sigma1_fail > sigma3
        assert safety > 0
        assert np.isfinite(sigma1_fail)
        assert np.isfinite(safety)

    def test_mohr_coulomb_array(self):
        """Test Mohr-Coulomb with arrays."""
        sigma1 = np.array([50.0, 60.0, 70.0])
        sigma3 = np.array([20.0, 25.0, 30.0])

        sigma1_fail, safety = mohr_coulomb_failure(
            sigma1, sigma3, cohesion=10.0, friction_angle=30.0
        )

        assert len(sigma1_fail) == 3
        assert len(safety) == 3
        assert np.all(sigma1_fail > sigma3)


class TestDruckerPragerFailure:
    """Tests for drucker_prager_failure."""

    def test_drucker_prager_basic(self):
        """Test basic Drucker-Prager calculation."""
        sqrt_J2_fail, safety = drucker_prager_failure(
            sigma1=50.0,
            sigma2=35.0,
            sigma3=20.0,
            cohesion=10.0,
            friction_angle=30.0,
        )

        assert sqrt_J2_fail > 0
        assert safety > 0
        assert np.isfinite(sqrt_J2_fail)
        assert np.isfinite(safety)


class TestHoekBrownFailure:
    """Tests for hoek_brown_failure."""

    def test_hoek_brown_basic(self):
        """Test basic Hoek-Brown calculation."""
        sigma1 = 50.0
        sigma3 = 20.0
        sigma1_fail, safety = hoek_brown_failure(
            sigma1=sigma1,
            sigma3=sigma3,
            ucs=50.0,
            mi=15.0,
            gsi=75.0,
            d=0.0,
        )

        assert sigma1_fail > sigma3
        assert safety > 0
        assert np.isfinite(sigma1_fail)
        assert np.isfinite(safety)

