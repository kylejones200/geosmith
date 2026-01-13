"""Geosmith ML: Confusion matrix visualization

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


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    title: str = "Confusion Matrix",
    figsize: tuple = (8, 7),
    normalize: bool = True,
):
    """
    Create a Matplotlib heatmap of the confusion matrix.

    Args:
        cm: Confusion matrix (numpy array)
        labels: List of class labels
        title: Title for the plot
        figsize: Figure size (width, height) in inches
        normalize: If True, show normalized values; if False, show counts

    Returns:
        Matplotlib figure object
    """
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    fig, ax = plt.subplots(figsize=figsize)

    # Normalize confusion matrix for display
    if normalize:
        cm_display = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_display = np.nan_to_num(cm_display)
        fmt = ".2f"
        cbar_label = "Normalized"
    else:
        cm_display = cm
        fmt = "d"
        cbar_label = "Count"

    # Create heatmap
    im = ax.imshow(cm_display, interpolation="nearest", cmap="Blues", aspect="auto")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, fontsize=10)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            if normalize:
                # Show percentage
                text = f"{cm_display[i, j]:.1%}"
            else:
                # Show count
                text = f"{cm[i, j]:d}"

            # Choose color based on background
            color = "white" if cm_display[i, j] > cm_display.max() / 2 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)

    # Labels and title
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)

    # signalplot handles spines automatically

    plt.tight_layout()
    return fig
