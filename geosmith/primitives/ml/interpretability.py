"""Geosmith ML: Model interpretability calculations (feature importance, SHAP)

Migrated from geosuite.ml.
Layer 2: Primitives - Pure operations.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import logging

logger = logging.getLogger(__name__)


try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None  # type: ignore

try:
    from sklearn.model_selection import BaseCrossValidator

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    BaseCrossValidator = None  # type: ignore

try:
    from shap import TreeExplainer, KernelExplainer

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    TreeExplainer = None  # type: ignore
    KernelExplainer = None  # type: ignore

try:
    from sklearn.inspection import partial_dependence as sklearn_partial_dependence

    SKLEARN_INSPECTION_AVAILABLE = True
except ImportError:
    SKLEARN_INSPECTION_AVAILABLE = False
    sklearn_partial_dependence = None  # type: ignore

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator if not args else decorator(args[0])


def get_feature_importance(
    model: Any, feature_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Extract feature importance from a trained model.

    Supports scikit-learn models with feature_importances_ attribute
    (Random Forest, Gradient Boosting, etc.).

    Parameters
    ----------
    model : Any
        Trained model with feature_importances_ attribute
    feature_names : list, optional
        Names of features. If None, uses indices.

    Returns
    -------
    pd.DataFrame
        DataFrame with feature names and importance scores

    Raises
    ------
    AttributeError
        If model doesn't have feature_importances_ attribute
    """
    # Try to get feature importances from model or pipeline
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        steps = (
            model.named_steps
            if hasattr(model, "named_steps")
            else dict(model.steps) if hasattr(model, "steps") else None
        )

        if steps is None:
            raise AttributeError(
                "Model does not have feature_importances_ attribute. "
                "Supported models: RandomForest, GradientBoosting, etc."
            )

        importances = next(
            (
                step_model.feature_importances_
                for step_model in steps.values()
                if hasattr(step_model, "feature_importances_")
            ),
            None,
        )

        if importances is None:
            raise AttributeError(
                "Model does not have feature_importances_ attribute. "
                "Supported models: RandomForest, GradientBoosting, etc."
            )

    # Get feature names
    if feature_names is None:
        feature_names = (
            list(model.feature_names_in_)
            if hasattr(model, "feature_names_in_")
            else (
                [f"Feature_{i}" for i in range(model.n_features_in_)]
                if hasattr(model, "n_features_in_")
                else [f"Feature_{i}" for i in range(len(importances))]
            )
        )

    # Create DataFrame
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    logger.info(f"Extracted feature importance for {len(importance_df)} features")

    return importance_df


def calculate_shap_values(
    model: Any,
    X: Union[np.ndarray, pd.DataFrame],
    max_samples: Optional[int] = 100,
    feature_names: Optional[List[str]] = None,
) -> Optional[np.ndarray]:
    """
    Calculate SHAP values for model interpretability.

    Requires shap library to be installed.

    Parameters
    ----------
    model : Any
        Trained model
    X : np.ndarray or pd.DataFrame
        Input features
    max_samples : int, optional
        Maximum number of samples to use (for large datasets)

    Returns
    -------
    np.ndarray or None
        SHAP values array, or None if shap is not available

    Raises
    ------
    ImportError
        If shap library is not installed
    """
    try:
        import shap
    except ImportError:
        raise ImportError(
            "shap library is required for SHAP values. "
            "Install with: pip install shap"
        )

    # Limit samples if needed
    if max_samples is not None and len(X) > max_samples:
        X_sample = (
            X.sample(n=max_samples, random_state=42)
            if isinstance(X, pd.DataFrame)
            else X[np.random.choice(len(X), max_samples, replace=False)]
        )
        logger.info(f"Using {max_samples} samples for SHAP calculation")
    else:
        X_sample = X

    # Create SHAP explainer
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # Handle multi-class case
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # Use first class for binary/multi-class

        logger.info(f"Calculated SHAP values: shape {shap_values.shape}")
        return shap_values

    except Exception as e:
        logger.warning(f"TreeExplainer failed, trying KernelExplainer: {e}")
        try:
            explainer = shap.KernelExplainer(model.predict, X_sample[:10])
            shap_values = explainer.shap_values(X_sample)
            return shap_values
        except Exception as e2:
            logger.error(f"Failed to calculate SHAP values: {e2}")
            return None
