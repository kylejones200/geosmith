"""Tests for surrogate models.

Tests fast emulation of expensive simulations using ML surrogates.
"""

import numpy as np
import pytest

from geosmith.objects.pointset import PointSet

# Check if dependencies are available
try:
    import sklearn  # noqa: F401
    from sklearn.ensemble import GradientBoostingRegressor  # noqa: F401

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost  # noqa: F401

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Only run tests if dependencies are available
pytestmark = pytest.mark.skipif(
    not SKLEARN_AVAILABLE, reason="scikit-learn not available"
)


@pytest.fixture
def sample_data():
    """Create synthetic sample data for testing."""
    np.random.seed(42)
    n_samples = 50
    coords = np.random.rand(n_samples, 2) * 100
    values = (
        coords[:, 0] * 0.1
        + coords[:, 1] * 0.15
        + np.sin(coords[:, 0] / 10) * 5
        + np.random.randn(n_samples) * 2
    )
    return PointSet(coordinates=coords), values


@pytest.fixture
def query_points():
    """Create query points for testing."""
    np.random.seed(42)
    query_coords = np.random.rand(100, 2) * 100
    return PointSet(coordinates=query_coords)


class TestSurrogateModel:
    """Tests for SurrogateModel class."""

    def test_surrogate_model_import(self):
        """Test that SurrogateModel can be imported."""
        try:
            from geosmith.primitives.surrogate import SurrogateModel

            assert SurrogateModel is not None
        except ImportError as e:
            pytest.skip(f"SurrogateModel not available: {e}")

    def test_surrogate_model_creation(self):
        """Test creating a SurrogateModel instance."""
        from geosmith.primitives.surrogate import SurrogateModel

        surrogate = SurrogateModel(
            model_type="gradient_boosting",
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
        )

        assert surrogate is not None
        assert surrogate.model_type == "gradient_boosting"
        assert surrogate.n_estimators == 50
        assert not surrogate.is_fitted

    def test_surrogate_model_creation_xgboost(self):
        """Test creating XGBoost surrogate model."""
        if not XGBOOST_AVAILABLE:
            pytest.skip("XGBoost not available")

        from geosmith.primitives.surrogate import SurrogateModel

        surrogate = SurrogateModel(
            model_type="xgboost",
            n_estimators=50,
            random_state=42,
        )

        assert surrogate.model_type == "xgboost"

    def test_surrogate_model_fit_basic(self, sample_data, query_points):
        """Test fitting a surrogate model with simple data."""
        from geosmith.primitives.surrogate import SurrogateModel

        samples, sample_values = sample_data

        # Create simple training data (simulated outputs)
        # For this test, we'll use a simple function as "simulation"
        def simple_simulation(sample_pts, query_pts):
            """Simple simulation function for testing."""
            # Concatenate coordinates as features
            X = np.vstack([sample_pts.coordinates, query_pts.coordinates])
            # Simple prediction: sum of coordinates
            y = X[:, 0] * 0.1 + X[:, 1] * 0.15
            return y

        # Generate training outputs
        training_outputs = simple_simulation(samples, query_points)

        # Train surrogate
        surrogate = SurrogateModel(
            model_type="gradient_boosting",
            n_estimators=50,
            max_depth=3,
            random_state=42,
            validation_split=0.2,
        )

        surrogate.fit(
            simulation_func=simple_simulation,
            training_inputs=[samples, query_points],
            training_outputs=training_outputs,
        )

        assert surrogate.is_fitted
        assert surrogate.model is not None
        assert surrogate.feature_scaler is not None
        assert surrogate.target_scaler is not None

    def test_surrogate_model_predict(self, sample_data, query_points):
        """Test prediction with fitted surrogate model."""
        from geosmith.primitives.surrogate import SurrogateModel

        samples, sample_values = sample_data

        # Create simple training data
        def simple_simulation(sample_pts, query_pts):
            X = np.vstack([sample_pts.coordinates, query_pts.coordinates])
            y = X[:, 0] * 0.1 + X[:, 1] * 0.15
            return y

        training_outputs = simple_simulation(samples, query_points)

        surrogate = SurrogateModel(
            model_type="gradient_boosting",
            n_estimators=50,
            max_depth=3,
            random_state=42,
            validation_split=0.0,  # No validation for speed
        )

        surrogate.fit(
            simulation_func=simple_simulation,
            training_inputs=[samples, query_points],
            training_outputs=training_outputs,
        )

        # Test prediction on new query points
        new_query = PointSet(coordinates=np.random.rand(20, 2) * 100)
        predictions = surrogate.predict(new_query)

        assert predictions is not None
        assert len(predictions) == len(new_query.coordinates)
        assert isinstance(predictions, np.ndarray)

    def test_surrogate_model_predict_not_fitted(self, query_points):
        """Test that prediction fails if model is not fitted."""
        from geosmith.primitives.surrogate import SurrogateModel

        surrogate = SurrogateModel(model_type="gradient_boosting", random_state=42)

        with pytest.raises(ValueError, match="must be fitted"):
            surrogate.predict(query_points)

    def test_surrogate_model_with_input_params(self, sample_data, query_points):
        """Test surrogate model with input parameters."""
        from geosmith.primitives.surrogate import SurrogateModel

        samples, sample_values = sample_data

        def simple_simulation(sample_pts, query_pts):
            X = np.vstack([sample_pts.coordinates, query_pts.coordinates])
            y = X[:, 0] * 0.1 + X[:, 1] * 0.15
            return y

        training_outputs = simple_simulation(samples, query_points)

        surrogate = SurrogateModel(
            model_type="gradient_boosting",
            n_estimators=50,
            max_depth=3,
            random_state=42,
            validation_split=0.0,
        )

        input_params = {"param1": 1.5, "param2": 2.0}

        surrogate.fit(
            simulation_func=simple_simulation,
            training_inputs=[samples, query_points],
            training_outputs=training_outputs,
            input_params=input_params,
        )

        # Test prediction with same params
        new_query = PointSet(coordinates=np.random.rand(10, 2) * 100)
        predictions = surrogate.predict(new_query, input_params=input_params)

        assert len(predictions) == len(new_query.coordinates)

    def test_surrogate_model_metrics(self, sample_data, query_points):
        """Test that metrics are computed during training."""
        from geosmith.primitives.surrogate import SurrogateModel

        samples, sample_values = sample_data

        def simple_simulation(sample_pts, query_pts):
            X = np.vstack([sample_pts.coordinates, query_pts.coordinates])
            y = X[:, 0] * 0.1 + X[:, 1] * 0.15
            return y

        training_outputs = simple_simulation(samples, query_points)

        surrogate = SurrogateModel(
            model_type="gradient_boosting",
            n_estimators=50,
            max_depth=3,
            random_state=42,
            validation_split=0.2,  # Enable validation
        )

        surrogate.fit(
            simulation_func=simple_simulation,
            training_inputs=[samples, query_points],
            training_outputs=training_outputs,
        )

        assert surrogate.metrics is not None
        assert hasattr(surrogate.metrics, "r2_score")
        assert hasattr(surrogate.metrics, "mae")
        assert hasattr(surrogate.metrics, "rmse")
        assert isinstance(surrogate.metrics.r2_score, float)
        assert isinstance(surrogate.metrics.mae, float)


class TestTrainSimulationEmulator:
    """Tests for train_simulation_emulator convenience function."""

    def test_train_simulation_emulator(self, sample_data, query_points):
        """Test train_simulation_emulator convenience function."""
        from geosmith.primitives.surrogate import train_simulation_emulator

        samples, sample_values = sample_data

        def simple_simulation(sample_pts, query_pts):
            X = np.vstack([sample_pts.coordinates, query_pts.coordinates])
            y = X[:, 0] * 0.1 + X[:, 1] * 0.15
            return y

        training_outputs = simple_simulation(samples, query_points)

        surrogate = train_simulation_emulator(
            simulation_func=simple_simulation,
            training_inputs=[samples, query_points],
            training_outputs=training_outputs,
            model_type="gradient_boosting",
            n_estimators=50,
            max_depth=3,
            random_state=42,
        )

        assert surrogate.is_fitted
        assert surrogate.model_type == "gradient_boosting"

        # Test prediction
        new_query = PointSet(coordinates=np.random.rand(10, 2) * 100)
        predictions = surrogate.predict(new_query)

        assert len(predictions) == len(new_query.coordinates)


class TestSurrogateTask:
    """Tests for SurrogateTask class."""

    def test_surrogate_task_import(self):
        """Test that SurrogateTask can be imported."""
        try:
            from geosmith.tasks.surrogatetask import SurrogateTask

            assert SurrogateTask is not None
        except ImportError as e:
            pytest.skip(f"SurrogateTask not available: {e}")

    def test_surrogate_task_creation(self):
        """Test creating a SurrogateTask instance."""
        from geosmith.tasks.surrogatetask import SurrogateTask

        task = SurrogateTask(
            model_type="gradient_boosting",
            n_estimators=50,
            random_state=42,
        )

        assert task is not None
        assert task.model_type == "gradient_boosting"

    def test_surrogate_task_train_and_predict(self, sample_data, query_points):
        """Test training and prediction using SurrogateTask."""
        from geosmith.tasks.surrogatetask import SurrogateTask

        samples, sample_values = sample_data

        def simple_simulation(sample_pts, query_pts):
            X = np.vstack([sample_pts.coordinates, query_pts.coordinates])
            y = X[:, 0] * 0.1 + X[:, 1] * 0.15
            return y

        training_outputs = simple_simulation(samples, query_points)

        task = SurrogateTask(
            model_type="gradient_boosting",
            n_estimators=50,
            max_depth=3,
            random_state=42,
        )

        # Train
        surrogate = task.train_emulator(
            simulation_func=simple_simulation,
            training_inputs=[samples, query_points],
            training_outputs=training_outputs,
        )

        assert surrogate.is_fitted

        # Predict
        new_query = PointSet(coordinates=np.random.rand(10, 2) * 100)
        predictions = task.predict(surrogate, new_query)

        assert len(predictions) == len(new_query.coordinates)

    def test_surrogate_task_validate(self, sample_data, query_points):
        """Test validation using SurrogateTask."""
        from geosmith.tasks.surrogatetask import SurrogateTask

        samples, sample_values = sample_data

        def simple_simulation(sample_pts, query_pts):
            X = np.vstack([sample_pts.coordinates, query_pts.coordinates])
            y = X[:, 0] * 0.1 + X[:, 1] * 0.15
            return y

        # Training data
        training_outputs = simple_simulation(samples, query_points)

        task = SurrogateTask(
            model_type="gradient_boosting",
            n_estimators=50,
            max_depth=3,
            random_state=42,
        )

        surrogate = task.train_emulator(
            simulation_func=simple_simulation,
            training_inputs=[samples, query_points],
            training_outputs=training_outputs,
        )

        # Test data (different query points)
        test_query = PointSet(coordinates=np.random.rand(30, 2) * 100)
        test_outputs = simple_simulation(samples, test_query)

        # Validate
        metrics = task.validate_emulator(
            surrogate,
            simulation_func=simple_simulation,
            test_inputs=[samples, test_query],
            test_outputs=test_outputs,
        )

        assert isinstance(metrics, dict)
        assert "r2_score" in metrics
        assert "mae" in metrics
        assert "rmse" in metrics
        assert isinstance(metrics["r2_score"], float)


class TestSurrogateModelAccuracy:
    """Tests for surrogate model accuracy on known functions."""

    def test_linear_function_accuracy(self):
        """Test that surrogate can accurately learn a linear function."""
        from geosmith.primitives.surrogate import train_simulation_emulator

        np.random.seed(42)
        n_samples = 50
        n_query = 100

        samples = PointSet(coordinates=np.random.rand(n_samples, 2) * 100)
        query_points = PointSet(coordinates=np.random.rand(n_query, 2) * 100)

        # Simple linear function: y = 2*x1 + 3*x2
        def linear_simulation(sample_pts, query_pts):
            X = np.vstack([sample_pts.coordinates, query_pts.coordinates])
            y = 2 * X[:, 0] + 3 * X[:, 1]
            return y

        training_outputs = linear_simulation(samples, query_points)

        surrogate = train_simulation_emulator(
            simulation_func=linear_simulation,
            training_inputs=[samples, query_points],
            training_outputs=training_outputs,
            model_type="gradient_boosting",
            n_estimators=100,
            max_depth=5,
            random_state=42,
            validation_split=0.2,
        )

        # Test on new points
        test_query = PointSet(coordinates=np.random.rand(50, 2) * 100)
        # linear_simulation returns outputs for both samples and test_query (vstacked)
        # but surrogate.predict only predicts on test_query, so we only need test_query outputs
        true_outputs = linear_simulation(samples, test_query)[len(samples.coordinates):]  # Only test_query portion
        predicted_outputs = surrogate.predict(test_query)

        # Should be very accurate for linear function
        r2 = 1 - np.sum((true_outputs - predicted_outputs) ** 2) / np.sum(
            (true_outputs - true_outputs.mean()) ** 2
        )

        # For a linear function, R² should be very high (>0.9)
        assert r2 > 0.9, f"R² score too low: {r2:.4f}"

        # MAE should be small relative to mean
        mae = np.mean(np.abs(true_outputs - predicted_outputs))
        mean_value = np.abs(true_outputs).mean()
        relative_error = mae / (mean_value + 1e-10)

        assert relative_error < 0.1, f"Relative error too high: {relative_error:.4f}"

