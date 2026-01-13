"""Tests for variogram primitives."""

import numpy as np
import pytest

from geosmith import PointSet
from geosmith.primitives.variogram import (
    VariogramModel,
    compute_experimental_variogram,
    fit_variogram_model,
    predict_variogram,
)


class TestVariogramModel:
    """Tests for VariogramModel."""

    def test_valid_model(self):
        """Test creating valid VariogramModel."""
        model = VariogramModel(
            model_type="spherical",
            nugget=0.1,
            sill=2.0,
            range_param=5.0,
            partial_sill=1.9,
            r_squared=0.95,
        )
        assert model.model_type == "spherical"
        assert model.nugget == 0.1
        assert model.sill == 2.0

    def test_invalid_model_type(self):
        """Test that invalid model_type raises error."""
        with pytest.raises(ValueError, match="model_type"):
            VariogramModel(
                model_type="invalid",
                nugget=0.1,
                sill=2.0,
                range_param=5.0,
                partial_sill=1.9,
                r_squared=0.95,
            )

    def test_invalid_sill(self):
        """Test that sill < nugget raises error."""
        with pytest.raises(ValueError, match="sill.*nugget"):
            VariogramModel(
                model_type="spherical",
                nugget=2.0,
                sill=1.0,  # Less than nugget
                range_param=5.0,
                partial_sill=-1.0,
                r_squared=0.95,
            )


class TestComputeExperimentalVariogram:
    """Tests for compute_experimental_variogram."""

    def test_compute_variogram(self):
        """Test computing experimental variogram."""
        # Create sample data with spatial correlation
        np.random.seed(42)
        coords = np.random.rand(50, 2) * 100
        # Create values with spatial correlation
        values = coords[:, 0] + coords[:, 1] + np.random.randn(50) * 5

        points = PointSet(coordinates=coords)

        lags, semi_vars, n_pairs = compute_experimental_variogram(
            points, values, n_lags=10
        )

        assert len(lags) > 0
        assert len(semi_vars) > 0
        assert len(n_pairs) > 0
        assert len(lags) == len(semi_vars) == len(n_pairs)
        assert np.all(semi_vars >= 0)  # Semi-variance should be non-negative

    def test_insufficient_samples(self):
        """Test that insufficient samples raises error."""
        coords = np.array([[0, 0], [1, 1], [2, 2]])  # Only 3 points
        values = np.array([1, 2, 3])
        points = PointSet(coordinates=coords)

        with pytest.raises(ValueError, match="at least 10 samples"):
            compute_experimental_variogram(points, values)


class TestFitVariogramModel:
    """Tests for fit_variogram_model."""

    def test_fit_spherical_model(self):
        """Test fitting spherical variogram model."""
        # Create synthetic variogram data
        lags = np.linspace(0, 50, 20)
        # Spherical model with known parameters
        nugget = 0.1
        sill = 2.0
        range_param = 20.0

        # Generate semi-variances using spherical model
        semi_vars = np.zeros_like(lags)
        mask = lags < range_param
        h_scaled = lags[mask] / range_param
        semi_vars[mask] = nugget + (sill - nugget) * (
            1.5 * h_scaled - 0.5 * h_scaled**3
        )
        semi_vars[~mask] = sill

        # Add some noise
        semi_vars += np.random.randn(len(semi_vars)) * 0.1

        # Fit model
        model = fit_variogram_model(lags, semi_vars, model_type="spherical")

        assert model.model_type == "spherical"
        assert model.nugget >= 0
        assert model.sill >= model.nugget
        assert model.range_param > 0


class TestPredictVariogram:
    """Tests for predict_variogram."""

    def test_predict_variogram(self):
        """Test predicting variogram values."""
        model = VariogramModel(
            model_type="spherical",
            nugget=0.1,
            sill=2.0,
            range_param=5.0,
            partial_sill=1.9,
            r_squared=0.95,
        )

        distances = np.array([0, 1, 2, 5, 10])
        predicted = predict_variogram(model, distances)

        assert len(predicted) == len(distances)
        assert np.all(predicted >= 0)
        assert predicted[0] == model.nugget  # At distance 0, should be nugget
        assert predicted[-1] <= model.sill  # At large distance, should approach sill


