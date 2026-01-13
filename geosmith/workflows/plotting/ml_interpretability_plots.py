"""Geosmith ML: Model interpretability plotting

Migrated from geosuite.ml.
Layer 4: Workflows - Plotting and I/O.
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
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None  # type: ignore
    Figure = None  # type: ignore

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None  # type: ignore

try:
    import signalplot

    SIGNALPLOT_AVAILABLE = True
except ImportError:
    SIGNALPLOT_AVAILABLE = False
    signalplot = None  # type: ignore

# Import pure calculation functions from primitives
from geosmith.primitives.ml.interpretability import (
    calculate_shap_values,
    get_feature_importance,
)


def plot_feature_importance(
    model: Any,
    feature_names: Optional[List[str]] = None,
    top_n: Optional[int] = None,
    figsize: tuple = (8, 6),
) -> Figure:
    """
    Plot feature importance from a trained model.

    Parameters
    ----------
    model : Any
        Trained model with feature_importances_
    feature_names : list, optional
        Names of features
    top_n : int, optional
        Number of top features to show (if None, shows all)
    figsize : tuple, default (8, 6)
        Figure size

    Returns
    -------
    Figure
        Matplotlib figure
    """
    try:
        import signalplot

        signalplot.apply()
    except ImportError:
        pass

    importance_df = get_feature_importance(model, feature_names)

    if top_n is not None:
        importance_df = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=figsize)

    ax.barh(
        range(len(importance_df)), importance_df["importance"].values, align="center"
    )
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df["feature"].values)
    ax.set_xlabel("Feature Importance", fontsize=11)
    ax.set_title("Feature Importance", fontsize=12, pad=10)
    ax.invert_yaxis()

    plt.tight_layout()

    return fig


def plot_shap_summary(
    model: Any,
    X: Union[np.ndarray, pd.DataFrame],
    feature_names: Optional[List[str]] = None,
    max_samples: Optional[int] = 100,
    figsize: tuple = (10, 8),
) -> Optional[Figure]:
    """
    Create SHAP summary plot.

    Parameters
    ----------
    model : Any
        Trained model
    X : np.ndarray or pd.DataFrame
        Input features
    feature_names : list, optional
        Feature names
    max_samples : int, optional
        Maximum samples for SHAP calculation
    figsize : tuple, default (10, 8)
        Figure size

    Returns
    -------
    Figure or None
        Matplotlib figure, or None if shap is not available
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap library not available, skipping SHAP plot")
        return None

    shap_values = calculate_shap_values(model, X, max_samples=max_samples)

    if shap_values is None:
        return None

    # Limit X for plotting
    if max_samples is not None and len(X) > max_samples:
        X_plot = (
            X.sample(n=max_samples, random_state=42)
            if isinstance(X, pd.DataFrame)
            else X[np.random.choice(len(X), max_samples, replace=False)]
        )
    else:
        X_plot = X

    # Create SHAP summary plot
    fig = plt.figure(figsize=figsize)
    shap.summary_plot(shap_values, X_plot, feature_names=feature_names, show=False)

    return fig


def partial_dependence_plot(
    model: Any,
    X: Union[np.ndarray, pd.DataFrame],
    feature: Union[int, str],
    feature_names: Optional[List[str]] = None,
    n_points: int = 50,
    figsize: tuple = (8, 6),
) -> Figure:
    """Create partial dependence plot for a single feature.

    Parameters
    ----------
    model : Any
        Trained model
    X : np.ndarray or pd.DataFrame
        Training data
    feature : int or str
        Feature index or name
    feature_names : list, optional
        Feature names (if X is array)
    n_points : int, default 50
        Number of points to evaluate
    figsize : tuple, default (8, 6)
        Figure size

    Returns
    -------
    Figure
        Matplotlib figure
    """
    try:
        from sklearn.inspection import partial_dependence
    except ImportError:
        raise ImportError(
            "scikit-learn is required for partial dependence plots. "
            "Install with: pip install scikit-learn"
        )

    if SIGNALPLOT_AVAILABLE:
        signalplot.apply()

    # Get feature index
    if isinstance(feature, str):
        if isinstance(X, pd.DataFrame):
            feature_idx = X.columns.get_loc(feature)
        elif feature_names:
            feature_idx = feature_names.index(feature)
        else:
            raise ValueError(f"Feature '{feature}' not found")
    else:
        feature_idx = feature

    # Calculate partial dependence
    pd_result = partial_dependence(
        model, X, features=[feature_idx], grid_resolution=n_points
    )

    # Extract values
    feature_values = pd_result["grid_values"][0]
    pd_values = pd_result["average"][0]

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(feature_values, pd_values, linewidth=2)
    ax.set_xlabel(
        feature if isinstance(feature, str) else f"Feature {feature}", fontsize=11
    )
    ax.set_ylabel("Partial Dependence", fontsize=11)
    ax.set_title("Partial Dependence Plot", fontsize=12, pad=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig
