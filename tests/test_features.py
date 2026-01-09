"""Tests for spatial feature engineering.

Migrated from geosuite.mining.features.
"""

import numpy as np
import pytest

# Try to import scikit-learn for tests
try:
    from sklearn.neighbors import KDTree
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from geosmith.primitives.features import (
    build_block_model_features,
    build_spatial_features,
)


class TestBuildSpatialFeatures:
    """Tests for build_spatial_features."""

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
    def test_basic_features(self):
        """Test basic spatial feature building."""
        coords = np.array(
            [[100, 200, 1000], [150, 250, 1100], [200, 300, 1200], [250, 350, 1300]]
        )

        features = build_spatial_features(coords, return_scalers=False)

        assert isinstance(features, np.ndarray)
        assert features.ndim == 2
        assert features.shape[0] == len(coords)
        assert features.shape[1] > 0  # Should have features

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
    def test_with_scalers(self):
        """Test building features with scalers returned."""
        coords = np.array(
            [[100, 200, 1000], [150, 250, 1100], [200, 300, 1200], [250, 350, 1300]]
        )

        features, scalers = build_spatial_features(coords, return_scalers=True)

        assert isinstance(features, np.ndarray)
        assert isinstance(scalers, dict)
        assert "coords_scaler" in scalers
        assert isinstance(scalers["coords_scaler"], StandardScaler)

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
    def test_feature_components(self):
        """Test individual feature components."""
        coords = np.array(
            [[100, 200, 1000], [150, 250, 1100], [200, 300, 1200], [250, 350, 1300]]
        )

        # Test with all components
        features_all = build_spatial_features(
            coords,
            include_depth=True,
            include_polynomial=True,
            include_density=True,
        )

        # Test with minimal components
        features_min = build_spatial_features(
            coords,
            include_depth=False,
            include_polynomial=False,
            include_density=False,
        )

        # All features should have more columns than minimal
        assert features_all.shape[1] > features_min.shape[1]

        # Minimal should have at least normalized coordinates (3 columns)
        assert features_min.shape[1] >= 3

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
    def test_polynomial_features(self):
        """Test polynomial feature generation."""
        coords = np.array(
            [[100, 200, 1000], [150, 250, 1100], [200, 300, 1200], [250, 350, 1300]]
        )

        features_deg2 = build_spatial_features(coords, poly_degree=2)
        features_deg3 = build_spatial_features(coords, poly_degree=3)

        # Degree 3 should have more features than degree 2
        assert features_deg3.shape[1] > features_deg2.shape[1]

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
    def test_invalid_coordinates(self):
        """Test that invalid coordinates raise errors."""
        # Wrong shape
        coords_2d = np.array([[100, 200], [150, 250]])
        with pytest.raises(ValueError, match="3 columns"):
            build_spatial_features(coords_2d)

        # Empty array
        coords_empty = np.array([]).reshape(0, 3)
        with pytest.raises(ValueError, match="cannot be empty"):
            build_spatial_features(coords_empty)

    def test_sklearn_unavailable(self):
        """Test that missing scikit-learn raises ImportError."""
        if SKLEARN_AVAILABLE:
            pytest.skip("scikit-learn is available")

        coords = np.array(
            [[100, 200, 1000], [150, 250, 1100], [200, 300, 1200], [250, 350, 1300]]
        )

        with pytest.raises(ImportError, match="scikit-learn is required"):
            build_spatial_features(coords)


class TestBuildBlockModelFeatures:
    """Tests for build_block_model_features."""

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
    def test_basic_block_model_features(self):
        """Test building features for block model."""
        sample_coords = np.array(
            [[100, 200, 1000], [150, 250, 1100], [200, 300, 1200], [250, 350, 1300]]
        )
        grid_coords = np.array(
            [[125, 225, 1050], [175, 275, 1150], [225, 325, 1250]]
        )

        # Build features and scalers from samples
        _, scalers = build_spatial_features(sample_coords, return_scalers=True)

        # Build features for grid
        grid_features = build_block_model_features(
            grid_coords, sample_coords, scalers
        )

        assert isinstance(grid_features, np.ndarray)
        assert grid_features.ndim == 2
        assert grid_features.shape[0] == len(grid_coords)
        assert grid_features.shape[1] > 0

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
    def test_missing_scalers(self):
        """Test that missing scalers raise error."""
        sample_coords = np.array(
            [[100, 200, 1000], [150, 250, 1100], [200, 300, 1200], [250, 350, 1300]]
        )
        grid_coords = np.array([[125, 225, 1050], [175, 275, 1150]])

        with pytest.raises(ValueError, match="must contain 'coords_scaler'"):
            build_block_model_features(grid_coords, sample_coords, {})

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
    def test_invalid_inputs(self):
        """Test that invalid inputs raise errors."""
        sample_coords = np.array(
            [[100, 200, 1000], [150, 250, 1100], [200, 300, 1200], [250, 350, 1300]]
        )
        _, scalers = build_spatial_features(sample_coords, return_scalers=True)

        # Wrong shape for grid
        grid_coords_2d = np.array([[125, 225], [175, 275]])
        with pytest.raises(ValueError, match="3 columns"):
            build_block_model_features(grid_coords_2d, sample_coords, scalers)

        # Wrong shape for samples
        sample_coords_2d = np.array([[100, 200], [150, 250]])
        grid_coords = np.array([[125, 225, 1050], [175, 275, 1150]])
        with pytest.raises(ValueError, match="3 columns"):
            build_block_model_features(grid_coords, sample_coords_2d, scalers)

    def test_sklearn_unavailable(self):
        """Test that missing scikit-learn raises ImportError."""
        if SKLEARN_AVAILABLE:
            pytest.skip("scikit-learn is available")

        sample_coords = np.array(
            [[100, 200, 1000], [150, 250, 1100], [200, 300, 1200], [250, 350, 1300]]
        )
        grid_coords = np.array([[125, 225, 1050], [175, 275, 1150]])
        scalers = {"coords_scaler": StandardScaler()}

        with pytest.raises(ImportError, match="scikit-learn is required"):
            build_block_model_features(grid_coords, sample_coords, scalers)

