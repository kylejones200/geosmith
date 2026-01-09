"""Tests for permeability calculation primitives."""

import numpy as np
import pytest

from geosmith.primitives.petrophysics import (
    calculate_permeability_kozeny_carman,
    calculate_permeability_porosity_only,
    calculate_permeability_timur,
)


class TestCalculatePermeabilityKozenyCarman:
    """Tests for calculate_permeability_kozeny_carman."""

    def test_kozeny_carman_basic(self):
        """Test basic Kozeny-Carman calculation."""
        k = calculate_permeability_kozeny_carman(phi=0.25)
        assert k > 0
        assert np.isfinite(k)

    def test_kozeny_carman_with_saturation(self):
        """Test Kozeny-Carman with water saturation."""
        k = calculate_permeability_kozeny_carman(phi=0.25, sw=0.5)
        assert k > 0
        assert np.isfinite(k)

    def test_kozeny_carman_array(self):
        """Test Kozeny-Carman with arrays."""
        phi = np.array([0.15, 0.25, 0.35])
        k = calculate_permeability_kozeny_carman(phi)
        assert len(k) == 3
        assert np.all(k > 0)


class TestCalculatePermeabilityTimur:
    """Tests for calculate_permeability_timur."""

    def test_timur_basic(self):
        """Test basic Timur calculation."""
        k = calculate_permeability_timur(phi=0.25, sw=0.5)
        assert k > 0
        assert np.isfinite(k)

    def test_timur_array(self):
        """Test Timur with arrays."""
        phi = np.array([0.15, 0.25, 0.35])
        sw = np.array([0.5, 0.6, 0.7])
        k = calculate_permeability_timur(phi, sw)
        assert len(k) == 3
        assert np.all(k > 0)

    def test_timur_mismatched_lengths(self):
        """Test that mismatched lengths raise error."""
        with pytest.raises(ValueError, match="same length"):
            calculate_permeability_timur(
                phi=np.array([0.25, 0.30]), sw=np.array([0.5])
            )


class TestCalculatePermeabilityPorosityOnly:
    """Tests for calculate_permeability_porosity_only."""

    def test_porosity_only_basic(self):
        """Test basic porosity-only calculation."""
        k = calculate_permeability_porosity_only(phi=0.25)
        assert k > 0
        assert np.isfinite(k)

    def test_porosity_only_array(self):
        """Test porosity-only with arrays."""
        phi = np.array([0.15, 0.25, 0.35])
        k = calculate_permeability_porosity_only(phi)
        assert len(k) == 3
        assert np.all(k > 0)

