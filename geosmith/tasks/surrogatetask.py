"""Surrogate model tasks for fast simulation emulation.

Layer 3: Tasks - User intent translation.
Provides user-friendly interfaces for training and using surrogate models.

This module wraps primitive surrogate model operations in a user-friendly
task interface that translates high-level user intent into low-level operations.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from geosmith.objects.pointset import PointSet
from geosmith.primitives.surrogate import SurrogateModel, train_simulation_emulator

logger = logging.getLogger(__name__)


class SurrogateTask:
    """Task for training and using surrogate models.

    Translates user intent for fast simulation emulation into surrogate model operations.

    Example:
        >>> from geosmith.tasks.surrogatetask import SurrogateTask
        >>> from geosmith.primitives.simulation import sequential_gaussian_simulation
        >>>
        >>> task = SurrogateTask(model_type='xgboost')
        >>> 
        >>> # Train surrogate on expensive simulation
        >>> surrogate = task.train_emulator(
        ...     simulation_func=sequential_gaussian_simulation,
        ...     training_data=simulation_io_pairs,
        ... )
        >>>
        >>> # Fast prediction
        >>> fast_results = task.predict(surrogate, query_points)
    """

    def __init__(
        self,
        model_type: str = "xgboost",
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        random_state: Optional[int] = 42,
    ):
        """Initialize SurrogateTask.

        Args:
            model_type: Model type ('xgboost', 'lightgbm', 'gradient_boosting', 'random_forest').
                Default 'xgboost' for best performance on large datasets.
            n_estimators: Number of boosting rounds/trees. Higher = more capacity, slower training.
                Default 200 is a good balance for most use cases.
            max_depth: Maximum tree depth. Controls model complexity and overfitting.
                Default 6 is typically sufficient.
            learning_rate: Learning rate for boosting. Lower = more iterations needed, better generalization.
                Default 0.05 is conservative and works well with n_estimators=200.
            random_state: Random seed for reproducibility. Default 42.

        Raises:
            ImportError: If required ML libraries are not available.
        """
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

    def train_emulator(
        self,
        simulation_func: Callable,
        training_inputs: Union[PointSet, np.ndarray, List[PointSet]],
        training_outputs: Union[np.ndarray, List[np.ndarray]],
        input_params: Optional[Dict[str, Any]] = None,
        n_training_samples: Optional[int] = None,
    ) -> SurrogateModel:
        """Train surrogate model emulator.

        Args:
            simulation_func: Expensive simulation function to emulate.
            training_inputs: Training input data.
            training_outputs: Training output data (simulation results).
            input_params: Optional fixed simulation parameters.
            n_training_samples: Optional limit on training samples.

        Returns:
            Trained SurrogateModel.

        Example:
            >>> task = SurrogateTask()
            >>> surrogate = task.train_emulator(
            ...     simulation_func=sequential_gaussian_simulation,
            ...     training_inputs=[samples, query_points],
            ...     training_outputs=sim_results,
            ... )
        """
        logger.info(f"Training {self.model_type} surrogate model...")

        surrogate = SurrogateModel(
            model_type=self.model_type,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )

        surrogate.fit(
            simulation_func=simulation_func,
            training_inputs=training_inputs,
            training_outputs=training_outputs,
            input_params=input_params,
            n_training_samples=n_training_samples,
        )

        return surrogate

    def predict(
        self,
        surrogate: SurrogateModel,
        query_points: Union[PointSet, np.ndarray],
        input_params: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Fast prediction using surrogate model.

        Args:
            surrogate: Trained SurrogateModel.
            query_points: Query locations.
            input_params: Optional simulation parameters.

        Returns:
            Predicted values.

        Example:
            >>> predictions = task.predict(surrogate, query_points)
            >>> print(f"Fast predictions: {len(predictions)} values")
        """
        return surrogate.predict(query_points, input_params)

    def validate_emulator(
        self,
        surrogate: SurrogateModel,
        simulation_func: Callable,
        test_inputs: Union[PointSet, np.ndarray, List[PointSet]],
        test_outputs: Union[np.ndarray, List[np.ndarray]],
        input_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Validate surrogate model accuracy.

        Args:
            surrogate: Trained SurrogateModel.
            simulation_func: Original simulation function.
            test_inputs: Test input data.
            test_outputs: True simulation outputs.
            input_params: Optional simulation parameters.

        Returns:
            Dictionary with validation metrics.

        Example:
            >>> metrics = task.validate_emulator(
            ...     surrogate, simulation_func, test_inputs, test_outputs
            ... )
            >>> print(f"R²: {metrics['r2_score']:.4f}, MAE: {metrics['mae']:.4f}")
        """
        metrics = surrogate.validate(
            simulation_func=simulation_func,
            test_inputs=test_inputs,
            test_outputs=test_outputs,
            input_params=input_params,
        )

        return {
            "r2_score": metrics.r2_score,
            "mae": metrics.mae,
            "rmse": metrics.rmse,
            "max_error": metrics.max_error,
            "mean_relative_error": metrics.mean_relative_error,
            "speedup_factor": metrics.speedup_factor,
        }

    def __repr__(self) -> str:
        """String representation of SurrogateTask."""
        return (
            f"SurrogateTask(model_type='{self.model_type}', "
            f"n_estimators={self.n_estimators}, max_depth={self.max_depth}, "
            f"learning_rate={self.learning_rate})"
        )

