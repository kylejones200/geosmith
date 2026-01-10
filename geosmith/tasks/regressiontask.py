"""Regression task for petrophysical property prediction.

Migrated from geosuite.ml.regression.
Layer 3: Tasks - User intent translation.
"""

import logging
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from geosmith.objects.geotable import GeoTable
from geosmith.primitives.base import BaseEstimator

logger = logging.getLogger(__name__)

# Optional scikit-learn dependency
try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Lasso, Ridge
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    RandomForestRegressor = None  # type: ignore
    GradientBoostingRegressor = None  # type: ignore
    Ridge = None  # type: ignore
    Lasso = None  # type: ignore
    Pipeline = None  # type: ignore
    StandardScaler = None  # type: ignore
    train_test_split = None  # type: ignore
    r2_score = None  # type: ignore
    mean_squared_error = None  # type: ignore
    mean_absolute_error = None  # type: ignore
    cross_val_score = None  # type: ignore


class PermeabilityPredictor(BaseEstimator):
    """Predict permeability from well log data.

    Uses machine learning regression models to predict permeability
    from petrophysical properties and well logs.

    Example:
        >>> from geosmith.tasks.regressiontask import PermeabilityPredictor
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> # Create sample data
        >>> X_train = pd.DataFrame({
        ...     'GR': np.random.normal(60, 15, 100),
        ...     'RHOB': np.random.normal(2.5, 0.2, 100),
        ...     'NPHI': np.random.normal(0.2, 0.1, 100)
        ... })
        >>> y_train = np.random.lognormal(0, 1, 100)  # Permeability in mD
        >>>
        >>> predictor = PermeabilityPredictor(model_type='random_forest')
        >>> predictor.fit(X_train, y_train)
        >>> predictions = predictor.predict(X_train)
        >>> print(f"Mean predicted permeability: {predictions.mean():.2f} mD")
    """

    def __init__(
        self,
        model_type: str = "random_forest",
        test_size: float = 0.2,
        random_state: int = 42,
        **model_params: Any,
    ) -> None:
        """Initialize the permeability predictor.

        Args:
            model_type: Type of model ('random_forest', 'gradient_boosting', 'ridge', 'lasso'), default 'random_forest'.
            test_size: Fraction of data to use for testing, default 0.2.
            random_state: Random seed for reproducibility, default 42.
            **model_params: Additional parameters passed to the model.
        """
        super().__init__()
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for regression. "
                "Install with: pip install geosmith[ml] or pip install scikit-learn"
            )

        self.model_type = model_type
        self.test_size = test_size
        self.random_state = random_state
        self.model_params = model_params
        self.model: Optional[Pipeline] = None
        self.scaler: Optional[StandardScaler] = None
        self._estimator_type = "regressor"

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
    ) -> "PermeabilityPredictor":
        """Fit the permeability prediction model.

        Args:
            X: Training features (well log data).
            y: Training targets (permeability values).

        Returns:
            self for method chaining.

        Example:
            >>> from geosmith.tasks.regressiontask import PermeabilityPredictor
            >>> import pandas as pd
            >>> import numpy as np
            >>>
            >>> X = pd.DataFrame({'GR': [60, 80, 40], 'RHOB': [2.5, 2.3, 2.7]})
            >>> y = np.array([100, 50, 200])  # Permeability in mD
            >>> predictor = PermeabilityPredictor(model_type='random_forest', test_size=0.0)
            >>> predictor.fit(X, y)
            PermeabilityPredictor(...)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for regression. "
                "Install with: pip install scikit-learn"
            )

        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        # Split data if test_size > 0
        if self.test_size > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
        else:
            X_train, y_train = X, y
            X_test, y_test = None, None

        # Create model pipeline
        model_configs = {
            "random_forest": {
                "default_params": {
                    "n_estimators": 300,
                    "max_depth": None,
                    "random_state": self.random_state,
                },
                "regressor": RandomForestRegressor,
            },
            "gradient_boosting": {
                "default_params": {
                    "n_estimators": 200,
                    "max_depth": 5,
                    "random_state": self.random_state,
                },
                "regressor": GradientBoostingRegressor,
            },
            "ridge": {
                "default_params": {"alpha": 1.0, "random_state": self.random_state},
                "regressor": Ridge,
            },
            "lasso": {
                "default_params": {"alpha": 1.0, "random_state": self.random_state},
                "regressor": Lasso,
            },
        }

        if self.model_type not in model_configs:
            raise ValueError(
                f"Unsupported model_type: {self.model_type}. "
                f"Supported: {', '.join(model_configs.keys())}"
            )

        config = model_configs[self.model_type]
        default_params = {**config["default_params"], **self.model_params}
        self.model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", config["regressor"](**default_params)),
            ]
        )

        # Fit model
        self.model.fit(X_train, y_train)

        # Evaluate on test set if available
        if X_test is not None:
            y_pred = self.model.predict(X_test)
            metrics = {
                "r2": r2_score(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "mae": mean_absolute_error(y_test, y_pred),
            }
            logger.info(
                f"Fitted {self.model_type} model: "
                f"R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}, "
                f"MAE = {metrics['mae']:.4f}"
            )
        else:
            logger.info(f"Fitted {self.model_type} model on full dataset")

        self._fitted = True
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict permeability values.

        Args:
            X: Input features (well log data).

        Returns:
            Predicted permeability values (non-negative).

        Example:
            >>> from geosmith.tasks.regressiontask import PermeabilityPredictor
            >>> import pandas as pd
            >>> import numpy as np
            >>>
            >>> X_train = pd.DataFrame({'GR': [60, 80, 40], 'RHOB': [2.5, 2.3, 2.7]})
            >>> y_train = np.array([100, 50, 200])
            >>> predictor = PermeabilityPredictor(model_type='random_forest', test_size=0.0)
            >>> predictor.fit(X_train, y_train)
            >>>
            >>> X_test = pd.DataFrame({'GR': [70], 'RHOB': [2.4]})
            >>> predictions = predictor.predict(X_test)
            >>> print(f"Predicted permeability: {predictions[0]:.2f} mD")
        """
        if not self._fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        predictions = self.model.predict(X)

        # Ensure non-negative (permeability can't be negative)
        predictions = np.maximum(predictions, 0)

        return predictions

    def score(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]
    ) -> float:
        """Score the model using R².

        Args:
            X: Input features.
            y: True target values.

        Returns:
            R² score.
        """
        if not self._fitted or self.model is None:
            raise ValueError("Model must be fitted before scoring")

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        y_pred = self.predict(X)
        return float(r2_score(y, y_pred))


class PorosityPredictor(BaseEstimator):
    """Predict porosity from well log data.

    Similar to PermeabilityPredictor but for porosity prediction.
    Porosity values are clipped to [0, 1] range.

    Example:
        >>> from geosmith.tasks.regressiontask import PorosityPredictor
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> X_train = pd.DataFrame({
        ...     'GR': np.random.normal(60, 15, 100),
        ...     'RHOB': np.random.normal(2.5, 0.2, 100),
        ...     'NPHI': np.random.normal(0.2, 0.1, 100)
        ... })
        >>> y_train = np.random.beta(2, 8, 100)  # Porosity (0-1)
        >>>
        >>> predictor = PorosityPredictor(model_type='random_forest')
        >>> predictor.fit(X_train, y_train)
        >>> predictions = predictor.predict(X_train)
        >>> print(f"Mean predicted porosity: {predictions.mean():.3f}")
    """

    def __init__(
        self,
        model_type: str = "random_forest",
        test_size: float = 0.2,
        random_state: int = 42,
        **model_params: Any,
    ) -> None:
        """Initialize the porosity predictor.

        Args:
            model_type: Type of model to use, default 'random_forest'.
            test_size: Fraction of data for testing, default 0.2.
            random_state: Random seed, default 42.
            **model_params: Additional model parameters.
        """
        super().__init__()
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for regression. "
                "Install with: pip install geosmith[ml] or pip install scikit-learn"
            )

        self.model_type = model_type
        self.test_size = test_size
        self.random_state = random_state
        self.model_params = model_params
        self.model: Optional[Pipeline] = None
        self._estimator_type = "regressor"

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
    ) -> "PorosityPredictor":
        """Fit the porosity prediction model.

        Args:
            X: Training features.
            y: Training targets (porosity values, typically 0-1).

        Returns:
            self for method chaining.

        Example:
            >>> from geosmith.tasks.regressiontask import PorosityPredictor
            >>> import pandas as pd
            >>> import numpy as np
            >>>
            >>> X = pd.DataFrame({'GR': [60, 80, 40], 'RHOB': [2.5, 2.3, 2.7]})
            >>> y = np.array([0.2, 0.15, 0.25])  # Porosity
            >>> predictor = PorosityPredictor(model_type='random_forest', test_size=0.0)
            >>> predictor.fit(X, y)
            PorosityPredictor(...)
        """
        # Use PermeabilityPredictor implementation (same structure)
        predictor = PermeabilityPredictor(
            model_type=self.model_type,
            test_size=self.test_size,
            random_state=self.random_state,
            **self.model_params,
        )
        predictor.fit(X, y)
        self.model = predictor.model
        self._fitted = True

        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict porosity values.

        Args:
            X: Input features.

        Returns:
            Predicted porosity values (clipped to [0, 1]).

        Example:
            >>> from geosmith.tasks.regressiontask import PorosityPredictor
            >>> import pandas as pd
            >>> import numpy as np
            >>>
            >>> X_train = pd.DataFrame({'GR': [60, 80, 40], 'RHOB': [2.5, 2.3, 2.7]})
            >>> y_train = np.array([0.2, 0.15, 0.25])
            >>> predictor = PorosityPredictor(model_type='random_forest', test_size=0.0)
            >>> predictor.fit(X_train, y_train)
            >>>
            >>> X_test = pd.DataFrame({'GR': [70], 'RHOB': [2.4]})
            >>> predictions = predictor.predict(X_test)
            >>> print(f"Predicted porosity: {predictions[0]:.3f}")
        """
        if not self._fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        predictions = self.model.predict(X)

        # Clip to valid porosity range [0, 1]
        predictions = np.clip(predictions, 0, 1)

        return predictions

    def score(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]
    ) -> float:
        """Score the model using R².

        Args:
            X: Input features.
            y: True target values.

        Returns:
            R² score.
        """
        if not self._fitted or self.model is None:
            raise ValueError("Model must be fitted before scoring")

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        y_pred = self.predict(X)
        return float(r2_score(y, y_pred))

