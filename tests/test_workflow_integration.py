"""Integration tests for complete geostatistical workflows.

Tests end-to-end workflows combining multiple primitives and tasks.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.integration

from geosmith import PointSet
from geosmith.primitives.kriging import CoKriging, OrdinaryKriging
from geosmith.primitives.kriging_cv import (
    CrossValidationResult,
    leave_one_out_cross_validation,
)
from geosmith.primitives.simulation import sequential_gaussian_simulation
from geosmith.primitives.variogram import (
    VariogramModel,
    compute_experimental_cross_variogram,
    compute_experimental_variogram,
    fit_variogram_model,
)
from geosmith.tasks.blockmodeltask import (
    create_block_model_grid,
    create_rotated_block_model_grid,
    create_sub_blocked_grid,
    create_variable_block_size_grid,
)
from geosmith.workflows.geostatistics import GeostatisticalModel


class TestOreReserveEstimationWorkflow:
    """Integration test for complete ore reserve estimation workflow."""

    def test_complete_workflow(self):
        """Test complete workflow from drillholes to grade estimation."""
        # Create synthetic drillhole data
        np.random.seed(42)
        n_samples = 50
        coords = np.random.rand(n_samples, 3) * 1000
        # Create spatially correlated grades
        grades = (
            coords[:, 0] * 0.1
            + coords[:, 1] * 0.15
            + np.sin(coords[:, 0] / 100) * 10
            + np.random.randn(n_samples) * 5
        )
        grades = np.maximum(grades, 0)  # Ensure non-negative

        samples = PointSet(coordinates=coords)

        # Step 1: Compute experimental variogram
        lags, semi_vars, n_pairs = compute_experimental_variogram(
            samples, grades, n_lags=15
        )
        assert len(lags) > 0
        assert len(semi_vars) > 0

        # Step 2: Fit variogram model
        variogram_model = fit_variogram_model(lags, semi_vars, model_type="spherical")
        assert variogram_model.r_squared > 0

        # Step 3: Cross-validation
        cv_result = leave_one_out_cross_validation(
            samples, grades, variogram_model, kriging_type="ordinary"
        )
        assert isinstance(cv_result, CrossValidationResult)
        assert cv_result.mae > 0
        assert cv_result.rmse > 0

        # Step 4: Create block model
        grid_coords, grid_info = create_block_model_grid(
            coords, block_size_xy=25.0, block_size_z=10.0
        )
        assert len(grid_coords) > 0
        assert grid_info["n_blocks"] > 0

        # Step 5: Kriging estimation
        grid_points = PointSet(coordinates=grid_coords)
        kriging = OrdinaryKriging(variogram_model=variogram_model)
        kriging.fit(samples, grades)
        result = kriging.predict(grid_points, return_variance=True)

        assert len(result.predictions) == len(grid_coords)
        assert np.all(np.isfinite(result.predictions))
        assert np.all(result.variance >= 0)

        # Step 6: SGS for uncertainty (reduced realizations for faster testing)
        realizations = sequential_gaussian_simulation(
            samples,
            grades,
            grid_points,
            variogram_model,
            n_realizations=2,  # Reduced for faster CI
            random_seed=42,
        )
        assert realizations.shape == (2, len(grid_coords))
        assert np.all(np.isfinite(realizations))


class TestCoKrigingWorkflow:
    """Integration test for Co-Kriging workflow."""

    def test_co_kriging_workflow(self):
        """Test complete Co-Kriging workflow with primary and secondary variables."""
        np.random.seed(42)

        # Create primary variable (sparse, expensive to measure)
        n_primary = 20
        primary_coords = np.random.rand(n_primary, 2) * 1000
        primary_values = (
            primary_coords[:, 0] * 0.1
            + primary_coords[:, 1] * 0.15
            + np.random.randn(n_primary) * 5
        )

        # Create secondary variable (dense, cheap to measure)
        n_secondary = 50
        secondary_coords = np.random.rand(n_secondary, 2) * 1000
        secondary_values = (
            secondary_coords[:, 0] * 0.08
            + secondary_coords[:, 1] * 0.12
            + np.random.randn(n_secondary) * 4
        )

        primary_points = PointSet(coordinates=primary_coords)
        secondary_points = PointSet(coordinates=secondary_coords)

        # Compute variograms
        lags_p, semi_vars_p, _ = compute_experimental_variogram(
            primary_points, primary_values, n_lags=10
        )
        lags_s, semi_vars_s, _ = compute_experimental_variogram(
            secondary_points, secondary_values, n_lags=10
        )

        # Compute cross-variogram
        lags_cross, cross_semi_vars, _ = compute_experimental_cross_variogram(
            primary_points,
            primary_values,
            secondary_points,
            secondary_values,
            n_lags=10,
        )

        # Fit variogram models
        primary_variogram = fit_variogram_model(
            lags_p, semi_vars_p, model_type="spherical"
        )
        secondary_variogram = fit_variogram_model(
            lags_s, semi_vars_s, model_type="spherical"
        )
        cross_variogram = fit_variogram_model(
            lags_cross, cross_semi_vars, model_type="spherical"
        )

        # Fit Co-Kriging
        co_kriging = CoKriging(
            primary_variogram=primary_variogram,
            secondary_variogram=secondary_variogram,
            cross_variogram=cross_variogram,
        )
        co_kriging.fit(primary_points, primary_values, secondary_points, secondary_values)

        # Predict at target locations
        query_coords = np.random.rand(10, 2) * 1000
        query_points = PointSet(coordinates=query_coords)
        result = co_kriging.predict(query_points, return_variance=True)

        assert len(result.predictions) == 10
        assert np.all(np.isfinite(result.predictions))
        assert np.all(result.variance >= 0)


class TestUnifiedGeostatisticalWorkflow:
    """Integration test for unified GeostatisticalModel workflow."""

    def test_unified_workflow(self):
        """Test unified geostatistical workflow interface."""
        np.random.seed(42)
        n_samples = 40
        coords = np.random.rand(n_samples, 3) * 1000
        values = (
            coords[:, 0] * 0.1
            + coords[:, 1] * 0.15
            + np.random.randn(n_samples) * 5
        )

        samples = PointSet(coordinates=coords)

        # Create unified model
        model = GeostatisticalModel(
            data=samples,
            values=values,
            method="kriging",
            kriging_type="ordinary",
            validation="cross_validate",
        )

        # Create target grid
        grid_coords, _ = create_block_model_grid(
            coords, block_size_xy=50.0, block_size_z=20.0
        )
        grid_points = PointSet(coordinates=grid_coords)

        # Estimate
        results = model.estimate(grid_points)

        assert results.estimates is not None
        assert results.uncertainty is not None
        assert results.validation_metrics is not None
        assert len(results.estimates) == len(grid_coords)
        assert np.all(np.isfinite(results.estimates))


class TestBlockModelEnhancements:
    """Integration tests for enhanced block model features."""

    def test_rotated_block_model(self):
        """Test rotated block model creation."""
        np.random.seed(42)
        coords = np.random.rand(30, 3) * 1000

        grid_coords, grid_info = create_rotated_block_model_grid(
            coords,
            block_size_xy=25.0,
            block_size_z=10.0,
            rotation_angle=45.0,
        )

        assert len(grid_coords) > 0
        assert "rotation_angle" in grid_info
        assert grid_info["rotation_angle"] == 45.0

    def test_sub_blocked_grid(self):
        """Test sub-blocking functionality."""
        np.random.seed(42)
        coords = np.random.rand(20, 3) * 500

        # Create parent grid
        parent_grid, parent_info = create_block_model_grid(
            coords, block_size_xy=50.0, block_size_z=20.0
        )

        # Create sub-blocked grid
        sub_grid, sub_info = create_sub_blocked_grid(
            parent_grid, parent_info, sub_divisions=(2, 2, 2)
        )

        assert len(sub_grid) > len(parent_grid)
        assert sub_info["n_blocks"] > parent_info["n_blocks"]
        assert "sub_divisions" in sub_info

    def test_variable_block_size_grid(self):
        """Test variable block size grid creation."""
        np.random.seed(42)
        coords = np.random.rand(30, 3) * 1000

        # Define regions with different block sizes
        regions = [
            {
                "bounds": {
                    "x_min": 0,
                    "x_max": 500,
                    "y_min": 0,
                    "y_max": 500,
                    "z_min": 0,
                    "z_max": 500,
                },
                "block_size_xy": 20.0,
                "block_size_z": 10.0,
            },
            {
                "bounds": {
                    "x_min": 500,
                    "x_max": 1000,
                    "y_min": 500,
                    "y_max": 1000,
                    "z_min": 500,
                    "z_max": 1000,
                },
                "block_size_xy": 50.0,
                "block_size_z": 25.0,
            },
        ]

        grid_coords, grid_info = create_variable_block_size_grid(
            coords, block_size_regions=regions
        )

        assert len(grid_coords) > 0
        assert grid_info["variable_block_sizes"] is True
        assert grid_info["n_regions"] == 2


class TestDataQualityWorkflow:
    """Integration tests for data quality tools in workflows."""

    def test_quality_filtering_workflow(self):
        """Test workflow with data quality filtering."""
        pytest.importorskip("scipy")
        pytest.importorskip("sklearn")

        from geosmith.primitives.data_quality import (
            compute_quality_flags,
            filter_by_quality,
        )

        np.random.seed(42)
        n_samples = 50
        coords = np.random.rand(n_samples, 3) * 1000
        values = np.random.rand(n_samples) * 10

        # Introduce some outliers and missing values
        values[0] = 1000  # Outlier
        values[1] = np.nan  # Missing value

        samples = PointSet(coordinates=coords)

        # Compute quality flags
        quality_flags = compute_quality_flags(
            samples, values, outlier_method="isolation_forest"
        )

        assert np.sum(quality_flags.is_outlier) > 0
        assert np.sum(quality_flags.has_missing_value) > 0

        # Filter by quality
        filtered_samples, filtered_values = filter_by_quality(
            samples, values, quality_flags, min_quality_score=0.5
        )

        assert len(filtered_samples.coordinates) < len(samples.coordinates)
        assert np.all(np.isfinite(filtered_values))

        # Now run kriging on filtered data
        lags, semi_vars, _ = compute_experimental_variogram(
            filtered_samples, filtered_values, n_lags=10
        )
        variogram_model = fit_variogram_model(lags, semi_vars, model_type="spherical")

        kriging = OrdinaryKriging(variogram_model=variogram_model)
        kriging.fit(filtered_samples, filtered_values)

        # Create query points
        query_coords = np.random.rand(10, 3) * 1000
        query_points = PointSet(coordinates=query_coords)
        result = kriging.predict(query_points)

        assert len(result.predictions) == 10
        assert np.all(np.isfinite(result.predictions))

