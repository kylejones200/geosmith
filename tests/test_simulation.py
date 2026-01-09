"""Tests for simulation primitives."""

import numpy as np
import pytest

from geosmith import PointSet
from geosmith.primitives.simulation import (
    compute_exceedance_probability,
    compute_simulation_statistics,
    sequential_gaussian_simulation,
)
from geosmith.primitives.variogram import VariogramModel


class TestSequentialGaussianSimulation:
    """Tests for sequential_gaussian_simulation."""

    def test_sgs_basic(self):
        """Test basic SGS functionality."""
        # Create sample data
        np.random.seed(42)
        n_samples = 20
        coords = np.random.rand(n_samples, 2) * 100
        values = np.random.rand(n_samples) * 10

        samples = PointSet(coordinates=coords)
        query_coords = np.random.rand(10, 2) * 100
        query_points = PointSet(coordinates=query_coords)

        # Create variogram model
        variogram = VariogramModel(
            model_type="spherical",
            nugget=0.1,
            sill=2.0,
            range_param=50.0,
            partial_sill=1.9,
            r_squared=0.95,
        )

        # Run SGS
        realizations = sequential_gaussian_simulation(
            samples,
            values,
            query_points,
            variogram,
            n_realizations=5,
            random_seed=42,
        )

        assert realizations.shape == (5, 10)
        assert np.all(np.isfinite(realizations))

    def test_sgs_insufficient_samples(self):
        """Test that insufficient samples raises error."""
        samples = PointSet(coordinates=np.array([[0, 0], [1, 1]]))
        query_points = PointSet(coordinates=np.array([[0.5, 0.5]]))
        variogram = VariogramModel(
            model_type="spherical",
            nugget=0.1,
            sill=2.0,
            range_param=5.0,
            partial_sill=1.9,
            r_squared=0.95,
        )

        with pytest.raises(ValueError, match="at least 3 samples"):
            sequential_gaussian_simulation(
                samples, np.array([1, 2]), query_points, variogram
            )


class TestComputeExceedanceProbability:
    """Tests for compute_exceedance_probability."""

    def test_exceedance_probability(self):
        """Test exceedance probability calculation."""
        # Create simple realizations
        realizations = np.array(
            [
                [1.0, 2.0, 3.0],  # Realization 1
                [2.0, 3.0, 4.0],  # Realization 2
                [1.5, 2.5, 3.5],  # Realization 3
            ]
        )

        prob = compute_exceedance_probability(realizations, threshold=2.0)

        # First point: [1.0, 2.0, 1.5] - 0/3 above 2.0 (all <= 2.0)
        # Second point: [2.0, 3.0, 2.5] - 2/3 above 2.0 (2.0 <= 2.0, 3.0 > 2.0, 2.5 > 2.0)
        # Third point: [3.0, 4.0, 3.5] - 3/3 above 2.0 (all > 2.0)
        assert prob[0] == pytest.approx(0.0)
        assert prob[1] == pytest.approx(2.0 / 3.0)
        assert prob[2] == pytest.approx(1.0)


class TestComputeSimulationStatistics:
    """Tests for compute_simulation_statistics."""

    def test_simulation_statistics(self):
        """Test simulation statistics calculation."""
        realizations = np.array(
            [
                [1.0, 2.0, 3.0],
                [2.0, 3.0, 4.0],
                [1.5, 2.5, 3.5],
            ]
        )

        stats = compute_simulation_statistics(realizations)

        assert "mean" in stats
        assert "std" in stats
        assert "p10" in stats
        assert "p50" in stats
        assert "p90" in stats
        assert "min" in stats
        assert "max" in stats

        assert len(stats["mean"]) == 3
        assert stats["mean"][0] == pytest.approx(1.5)

