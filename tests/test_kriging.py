"""Tests for kriging primitives."""

import numpy as np
import pytest

from geosmith import PointSet
from geosmith.primitives.kriging import KrigingResult, OrdinaryKriging
from geosmith.primitives.variogram import VariogramModel


class TestOrdinaryKriging:
    """Tests for OrdinaryKriging."""

    def test_fit_and_predict(self):
        """Test fitting and prediction."""
        # Create sample data
        coords = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        points = PointSet(coordinates=coords)

        # Create variogram model
        variogram = VariogramModel(
            model_type="spherical",
            nugget=0.1,
            sill=2.0,
            range_param=5.0,
            partial_sill=1.9,
            r_squared=0.95,
        )

        # Fit kriging
        kriging = OrdinaryKriging(variogram_model=variogram)
        kriging.fit(points, values)

        assert kriging.is_fitted

        # Predict at new locations
        query_coords = np.array([[1.5, 1.5], [2.5, 2.5]])
        query_points = PointSet(coordinates=query_coords)
        result = kriging.predict(query_points, return_variance=True)

        assert isinstance(result, KrigingResult)
        assert len(result.predictions) == 2
        assert len(result.variance) == 2
        assert np.all(result.variance >= 0)  # Variance should be non-negative

    def test_not_fitted_error(self):
        """Test that predict raises error if not fitted."""
        variogram = VariogramModel(
            model_type="spherical",
            nugget=0.1,
            sill=2.0,
            range_param=5.0,
            partial_sill=1.9,
            r_squared=0.95,
        )

        kriging = OrdinaryKriging(variogram_model=variogram)
        query_points = PointSet(coordinates=np.array([[1, 1]]))

        with pytest.raises(ValueError, match="not fitted"):
            kriging.predict(query_points)


