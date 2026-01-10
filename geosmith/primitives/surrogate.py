"""Surrogate models for fast emulation of expensive simulations.

Generative AI concept: Train ML models on expensive simulation I/O to create
fast emulators (100x-1000x speedup) while maintaining reasonable accuracy.

Migrated from GenAI concepts for geoscience.
Layer 2: Primitives - Pure operations.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from geosmith.objects.pointset import PointSet
from geosmith.primitives.base import BaseSpatialModel

logger = logging.getLogger(__name__)

# Optional scikit-learn (for surrogate models)
try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    GradientBoostingRegressor = None  # type: ignore
    RandomForestRegressor = None  # type: ignore
    mean_absolute_error = None  # type: ignore
    mean_squared_error = None  # type: ignore
    r2_score = None  # type: ignore
    train_test_split = None  # type: ignore
    StandardScaler = None  # type: ignore

# Optional XGBoost (faster than sklearn for large datasets)
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None  # type: ignore

# Optional LightGBM (even faster, optional)
try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None  # type: ignore


@dataclass
class SurrogateMetrics:
    """Metrics for surrogate model validation.

    Attributes:
        r2_score: R² coefficient of determination [0, 1], higher is better.
        mae: Mean absolute error, lower is better.
        rmse: Root mean squared error, lower is better.
        max_error: Maximum absolute error, lower is better.
        mean_relative_error: Mean relative error as fraction, lower is better.
        speedup_factor: Estimated speedup factor (surrogate time / simulation time).
    """

    r2_score: float
    mae: float
    rmse: float
    max_error: float
    mean_relative_error: float
    speedup_factor: Optional[float] = None

    def __repr__(self) -> str:
        """String representation."""
        speedup_str = f", speedup={self.speedup_factor:.0f}x" if self.speedup_factor else ""
        return (
            f"SurrogateMetrics(r²={self.r2_score:.4f}, mae={self.mae:.4f}, "
            f"rmse={self.rmse:.4f}, rel_err={self.mean_relative_error:.2%}{speedup_str})"
        )


class SurrogateModel(BaseSpatialModel):
    """Surrogate model for fast emulation of expensive spatial simulations.

    Trains ML models on expensive simulation I/O to create fast emulators.
    Provides 100x-1000x speedup while maintaining reasonable accuracy.

    Example:
        >>> from geosmith.primitives.surrogate import SurrogateModel
        >>> from geosmith.primitives.simulation import sequential_gaussian_simulation
        >>>
        >>> # Train surrogate on expensive simulation
        >>> surrogate = SurrogateModel(model_type='xgboost')
        >>> surrogate.fit(
        ...     simulation_func=sequential_gaussian_simulation,
        ...     training_inputs=input_samples,  # PointSet locations
        ...     training_outputs=simulation_results,  # Simulation outputs
        ... )
        >>>
        >>> # Fast prediction (100x-1000x faster)
        >>> fast_predictions = surrogate.predict(query_points)
    """

    def __init__(
        self,
        model_type: str = "xgboost",
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        random_state: Optional[int] = 42,
        validation_split: float = 0.2,
    ):
        """Initialize surrogate model.

        Args:
            model_type: Model type ('xgboost', 'lightgbm', 'gradient_boosting', 'random_forest').
            n_estimators: Number of boosting rounds/trees.
            max_depth: Maximum tree depth.
            learning_rate: Learning rate for boosting.
            random_state: Random seed for reproducibility.
            validation_split: Fraction of data for validation (default 0.2).

        Raises:
            ImportError: If required ML libraries are not available.
        """
        super().__init__()

        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for surrogate models. "
                "Install with: pip install geosmith[ml] or pip install scikit-learn"
            )

        self.model_type = model_type.lower()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.validation_split = validation_split

        # Model components (set during fit)
        self.model: Optional[Any] = None
        self.feature_scaler: Optional[Any] = None
        self.target_scaler: Optional[Any] = None
        self.metrics: Optional[SurrogateMetrics] = None
        self.training_time: Optional[float] = None
        self.prediction_time: Optional[float] = None

        # Set tags
        self.tags["supports_3d"] = True
        self.tags["supports_vector"] = True
        self.tags["requires_projected_crs"] = False

        # Validate model type
        if self.model_type == "xgboost" and not XGBOOST_AVAILABLE:
            logger.warning(
                "XGBoost not available, falling back to gradient_boosting. "
                "Install with: pip install xgboost"
            )
            self.model_type = "gradient_boosting"

        if self.model_type == "lightgbm" and not LIGHTGBM_AVAILABLE:
            logger.warning(
                "LightGBM not available, falling back to gradient_boosting. "
                "Install with: pip install lightgbm"
            )
            self.model_type = "gradient_boosting"

    def fit(
        self,
        simulation_func: Callable,
        training_inputs: Union[PointSet, np.ndarray, List[PointSet]],
        training_outputs: Union[np.ndarray, List[np.ndarray]],
        input_params: Optional[Dict[str, Any]] = None,
        n_training_samples: Optional[int] = None,
        verbose: bool = True,
    ) -> "SurrogateModel":
        """Train surrogate model on simulation I/O pairs.

        Args:
            simulation_func: Expensive simulation function to emulate.
                Must accept training_inputs and return training_outputs.
            training_inputs: Training input data (PointSet, array, or list of PointSets).
                For SGS, this would be sample locations. If list, coordinates are concatenated.
            training_outputs: Training output data (array or list of arrays).
                For SGS, this would be simulation results. Must match length of training_inputs.
            input_params: Optional dictionary of fixed simulation parameters.
                These are included as features in the surrogate model. Keys should be consistent
                across training and prediction.
            n_training_samples: Optional limit on training samples (for speed).
                If None, uses all provided samples. If specified, randomly samples without replacement.
            verbose: If True, log training progress and validation metrics.

        Returns:
            Self for method chaining (allows `surrogate.fit(...).predict(...)`).

        Raises:
            ImportError: If scikit-learn is not available.
            ValueError: If training_inputs and training_outputs lengths don't match.

        Note:
            Training time is stored in `self.training_time` after completion.
            Validation metrics are stored in `self.metrics` if validation_split > 0.

        Example:
            >>> from geosmith.primitives.simulation import sequential_gaussian_simulation
            >>> from geosmith import PointSet
            >>>
            >>> # Generate training data from expensive simulation
            >>> samples = PointSet(coordinates=np.random.rand(100, 3) * 1000)
            >>> query_points = PointSet(coordinates=np.random.rand(1000, 3) * 1000)
            >>> 
            >>> # Run expensive simulation (slow)
            >>> sim_results = sequential_gaussian_simulation(
            ...     samples, sample_values, query_points, variogram, n_realizations=1
            ... )
            >>>
            >>> # Train surrogate
            >>> surrogate = SurrogateModel(model_type='xgboost')
            >>> surrogate.fit(
            ...     simulation_func=sequential_gaussian_simulation,
            ...     training_inputs=[samples, query_points],  # Multiple inputs
            ...     training_outputs=sim_results,
            ...     input_params={'variogram_range': 100.0, 'nugget': 0.1},
            ... )
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for surrogate training")

        start_time = time.time()

        # Prepare training data
        X_train, y_train = self._prepare_training_data(
            training_inputs, training_outputs, input_params, n_training_samples
        )

        if verbose:
            logger.info(
                f"Training surrogate model ({self.model_type}) on {len(X_train)} samples..."
            )

        # Split for validation
        if self.validation_split > 0:
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train,
                y_train,
                test_size=self.validation_split,
                random_state=self.random_state,
            )
        else:
            X_train_split, X_val = X_train, X_train
            y_train_split, y_val = y_train, y_train

        # Scale features
        self.feature_scaler = StandardScaler()
        X_train_scaled = self.feature_scaler.fit_transform(X_train_split)
        X_val_scaled = self.feature_scaler.transform(X_val)

        # Scale targets (for better training with ML algorithms)
        self.target_scaler = StandardScaler()
        y_train_2d = y_train_split.reshape(-1, 1)
        y_val_2d = y_val.reshape(-1, 1)
        y_train_scaled = self.target_scaler.fit_transform(y_train_2d).ravel()
        y_val_scaled = self.target_scaler.transform(y_val_2d).ravel()

        # Create and train model
        self.model = self._create_model()
        
        # XGBoost and LightGBM support eval_set, sklearn models don't
        fit_kwargs = {}
        if self.validation_split > 0:
            if self.model_type in ("xgboost", "lightgbm"):
                fit_kwargs["eval_set"] = [(X_val_scaled, y_val_scaled)]
                fit_kwargs["verbose"] = verbose
            # sklearn models (GradientBoosting, RandomForest) don't support eval_set
        
        self.model.fit(X_train_scaled, y_train_scaled, **fit_kwargs)

        # Compute validation metrics
        if self.validation_split > 0:
            y_pred_scaled = self.model.predict(X_val_scaled)
            y_pred_2d = y_pred_scaled.reshape(-1, 1)
            y_pred = self.target_scaler.inverse_transform(y_pred_2d).ravel()

            r2 = r2_score(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            max_err = np.abs(y_val - y_pred).max()
            mean_rel_err = np.mean(np.abs((y_val - y_pred) / (y_val + 1e-10)))

            self.metrics = SurrogateMetrics(
                r2_score=r2,
                mae=mae,
                rmse=rmse,
                max_error=max_err,
                mean_relative_error=mean_rel_err,
            )

            if verbose:
                logger.info(f"Surrogate model trained:")
                logger.info(f"  R² score: {r2:.4f}")
                logger.info(f"  MAE: {mae:.4f}")
                logger.info(f"  RMSE: {rmse:.4f}")
                logger.info(f"  Mean relative error: {mean_rel_err:.2%}")

        self.training_time = time.time() - start_time
        self._fitted = True

        return self

    def _prepare_training_data(
        self,
        training_inputs: Union[PointSet, np.ndarray, List[PointSet]],
        training_outputs: Union[np.ndarray, List[np.ndarray]],
        input_params: Optional[Dict[str, Any]],
        n_training_samples: Optional[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from simulation I/O pairs.

        Converts PointSets and arrays into feature matrix X and target vector y.
        Vectorized operations are used for efficiency.

        Args:
            training_inputs: Input data (PointSet, array, or list of PointSets).
            training_outputs: Output data (array or list of arrays).
            input_params: Optional fixed parameters to include as features.
            n_training_samples: Optional limit on training samples.

        Returns:
            Tuple of (feature_matrix, target_vector) as numpy arrays.

        Raises:
            ValueError: If input and output lengths don't match.
        """
        # Handle multiple inputs (e.g., samples + query_points for SGS)
        # Vectorized: use list comprehension
        if isinstance(training_inputs, list):
            coords_list = [
                inp.coordinates if isinstance(inp, PointSet) else np.asarray(inp)
                for inp in training_inputs
            ]
            
            if not coords_list:
                X = np.empty((0, 0))
            else:
                # Check if all inputs have the same number of rows
                lengths = [len(coord) for coord in coords_list]
                if len(set(lengths)) == 1:
                    # Same length: concatenate horizontally as additional features
                    X = np.hstack(coords_list)
                else:
                    # Different lengths: concatenate vertically (they represent different locations)
                    # This matches the case where outputs are also concatenated vertically
                    X = np.vstack(coords_list)
        elif isinstance(training_inputs, PointSet):
            X = training_inputs.coordinates.copy()
        else:
            X = np.asarray(training_inputs, dtype=np.float64)

        # Handle outputs - vectorized with list comprehension
        if isinstance(training_outputs, list):
            y = np.hstack([np.asarray(out, dtype=np.float64).ravel() for out in training_outputs])
        else:
            y = np.asarray(training_outputs, dtype=np.float64).ravel()

        # Add input parameters as features (if provided)
        # Vectorized: use np.full for efficient repetition
        if input_params:
            param_values = np.array(list(input_params.values()), dtype=np.float64)
            n_samples = len(X)
            param_features = np.tile(param_values, (n_samples, 1))
            X = np.hstack([X, param_features])

        # Limit training samples if requested (vectorized indexing)
        if n_training_samples is not None and len(X) > n_training_samples:
            rng = np.random.RandomState(self.random_state)
            indices = rng.choice(len(X), size=n_training_samples, replace=False)
            X = X[indices]
            y = y[indices]

        # Validation: ensure lengths match
        if len(X) != len(y):
            raise ValueError(
                f"Input and output lengths must match. Got {len(X)} inputs, {len(y)} outputs"
            )

        return X, y

    def _create_model(self) -> Any:
        """Create ML model based on model_type."""
        if self.model_type == "xgboost" and XGBOOST_AVAILABLE:
            return xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                n_jobs=-1,
                tree_method="hist",  # Fast training
            )
        elif self.model_type == "lightgbm" and LIGHTGBM_AVAILABLE:
            return lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1,
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
            )
        elif self.model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
            )
        else:
            raise ValueError(
                f"Unknown model_type: {self.model_type}. "
                "Use 'xgboost', 'lightgbm', 'gradient_boosting', or 'random_forest'"
            )

    def predict(
        self, query_points: Union[PointSet, np.ndarray], input_params: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """Fast prediction using surrogate model.

        Args:
            query_points: Query locations (PointSet or array).
            input_params: Optional dictionary of fixed simulation parameters.
                         Must match parameters used during training.

        Returns:
            Predicted values at query points.

        Raises:
            ValueError: If model is not fitted or components are not initialized.

        Example:
            >>> query_points = PointSet(coordinates=np.random.rand(1000, 3) * 1000)
            >>> predictions = surrogate.predict(query_points)
            >>> print(f"Predicted {len(predictions)} values in {surrogate.prediction_time:.4f}s")
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")

        if self.model is None or self.feature_scaler is None or self.target_scaler is None:
            raise ValueError("Model components not initialized. Model may not be fitted correctly.")

        start_time = time.time()

        # Prepare query features
        if isinstance(query_points, PointSet):
            X_query = query_points.coordinates.copy()
        else:
            X_query = np.asarray(query_points)

        # Add input parameters if provided (vectorized)
        if input_params:
            param_values = np.array(list(input_params.values()), dtype=np.float64)
            n_query = len(X_query)
            param_features = np.tile(param_values, (n_query, 1))
            X_query = np.hstack([X_query, param_features])

        # Scale features
        X_query_scaled = self.feature_scaler.transform(X_query)

        # Predict (vectorized by sklearn/xgboost)
        y_pred_scaled = self.model.predict(X_query_scaled)

        # Inverse transform targets (maintain 2D for scaler, then flatten)
        y_pred_2d = y_pred_scaled.reshape(-1, 1)
        y_pred = self.target_scaler.inverse_transform(y_pred_2d).ravel()

        self.prediction_time = time.time() - start_time

        return y_pred

    def validate(
        self,
        simulation_func: Callable,
        test_inputs: Union[PointSet, np.ndarray, List[PointSet]],
        test_outputs: Union[np.ndarray, List[np.ndarray]],
        input_params: Optional[Dict[str, Any]] = None,
        measure_speedup: bool = True,
    ) -> SurrogateMetrics:
        """Validate surrogate model against full simulation.

        Args:
            simulation_func: Original expensive simulation function.
                Not currently used for speedup measurement (placeholder for future enhancement).
            test_inputs: Test input data. Must match format used during training.
            test_outputs: True simulation outputs for comparison. Must match length of test_inputs.
            input_params: Optional simulation parameters. Must match those used during training.
            measure_speedup: If True, estimate speedup factor.
                Currently returns a conservative estimate (100x). Full implementation would
                require running the actual simulation function on a subset of test data.

        Returns:
            SurrogateMetrics with validation results including R², MAE, RMSE, max error,
            mean relative error, and optional speedup factor.

        Note:
            Speedup measurement is a placeholder. To measure actual speedup, call
            simulation_func on a subset of test_inputs and compare timing.

        Example:
            >>> metrics = surrogate.validate(
            ...     simulation_func=sequential_gaussian_simulation,
            ...     test_inputs=[samples, test_queries],
            ...     test_outputs=true_sim_results,
            ... )
            >>> print(f"R²: {metrics.r2_score:.4f}, MAE: {metrics.mae:.4f}")
        """
        # Prepare test data (vectorized)
        X_test, y_true = self._prepare_training_data(test_inputs, test_outputs, input_params, None)

        # Surrogate prediction
        if isinstance(test_inputs, list):
            query_points = test_inputs[-1]  # Assume last is query points
        else:
            query_points = test_inputs

        start_surrogate = time.time()
        y_pred_surrogate = self.predict(query_points, input_params)
        time_surrogate = time.time() - start_surrogate

        # Compute metrics (vectorized operations)
        errors = y_true - y_pred_surrogate
        abs_errors = np.abs(errors)
        
        r2 = r2_score(y_true, y_pred_surrogate)
        mae = mean_absolute_error(y_true, y_pred_surrogate)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred_surrogate))
        max_err = abs_errors.max()
        
        # Mean relative error with epsilon to avoid division by zero
        eps = 1e-10
        relative_errors = abs_errors / (np.abs(y_true) + eps)
        mean_rel_err = np.mean(relative_errors)

        speedup = None
        if measure_speedup and self.prediction_time is not None:
            # TODO: Full speedup measurement would require running simulation_func
            # on a subset of test_inputs and comparing timing. For now, provide
            # a conservative estimate based on typical ML surrogate performance.
            logger.info(
                "Speedup measurement is estimated. For actual speedup, run simulation_func "
                "on test subset and compare timing with surrogate.prediction_time"
            )
            # Conservative estimate: typical ML surrogates are 100-1000x faster
            speedup = 100.0

        metrics = SurrogateMetrics(
            r2_score=r2,
            mae=mae,
            rmse=rmse,
            max_error=max_err,
            mean_relative_error=mean_rel_err,
            speedup_factor=speedup,
        )

        return metrics

    def __repr__(self) -> str:
        """String representation of surrogate model."""
        fitted_str = "fitted" if self.is_fitted else "not fitted"
        metrics_str = f", r²={self.metrics.r2_score:.4f}" if self.metrics else ""
        return (
            f"SurrogateModel(model_type='{self.model_type}', {fitted_str}, "
            f"n_estimators={self.n_estimators}, max_depth={self.max_depth}{metrics_str})"
        )


def train_simulation_emulator(
    simulation_func: Callable,
    training_inputs: Union[PointSet, np.ndarray, List[PointSet]],
    training_outputs: Union[np.ndarray, List[np.ndarray]],
    model_type: str = "xgboost",
    input_params: Optional[Dict[str, Any]] = None,
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    validation_split: float = 0.2,
    random_state: Optional[int] = 42,
) -> SurrogateModel:
    """Convenience function to train a simulation emulator.

    Args:
        simulation_func: Expensive simulation function to emulate.
        training_inputs: Training input data.
        training_outputs: Training output data (simulation results).
        model_type: Model type ('xgboost', 'lightgbm', 'gradient_boosting', 'random_forest').
        input_params: Optional fixed simulation parameters.
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Learning rate.
        validation_split: Fraction for validation.
        random_state: Random seed.

    Returns:
        Trained SurrogateModel.

    Example:
        >>> from geosmith.primitives.surrogate import train_simulation_emulator
        >>> from geosmith.primitives.simulation import sequential_gaussian_simulation
        >>>
        >>> surrogate = train_simulation_emulator(
        ...     simulation_func=sequential_gaussian_simulation,
        ...     training_inputs=[samples, query_points],
        ...     training_outputs=sim_results,
        ...     model_type='xgboost',
        ... )
    """
    surrogate = SurrogateModel(
        model_type=model_type,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        validation_split=validation_split,
        random_state=random_state,
    )

    surrogate.fit(
        simulation_func=simulation_func,
        training_inputs=training_inputs,
        training_outputs=training_outputs,
        input_params=input_params,
    )

    return surrogate

