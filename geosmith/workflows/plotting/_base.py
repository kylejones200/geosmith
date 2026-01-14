"""Base utilities for plotting modules.

Common imports, matplotlib setup, helper functions, and constants shared
across all plotting modules.

Layer 4: Workflows - Public entry points with plotting.
"""

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# Optional matplotlib dependency
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None  # type: ignore
    Figure = None  # type: ignore
    Axes = None  # type: ignore
    mpatches = None  # type: ignore
    logger.warning(
        "matplotlib not available. Plotting functions require matplotlib. "
        "Install with: pip install geosmith[viz] or pip install matplotlib"
    )


def _is_plotsmith_available() -> bool:
    """Check if PlotSmith is available.

    Returns:
        True if PlotSmith can be imported, False otherwise.
    """
    try:
        import plotsmith  # noqa: F401
        return True
    except ImportError:
        return False


def _apply_plotsmith_style() -> None:
    """Apply PlotSmith styling if available."""
    if _is_plotsmith_available():
        try:
            import plotsmith
            if hasattr(plotsmith, "apply"):
                plotsmith.apply()
            elif hasattr(plotsmith, "style"):
                plotsmith.style.apply()
        except (ImportError, AttributeError):
            pass  # Fall back to default matplotlib


# Default color schemes for facies
FACIES_COLORS = {
    "Sand": "#FFFF00",
    "Shale": "#808080",
    "Siltstone": "#C8C896",
    "Carbonate": "#0000FF",
    "Limestone": "#87CEEB",
    "Dolomite": "#FFB6C1",
    "Coal": "#000000",
    "Clean_Sand": "#FFFF00",
    "Shaly_Sand": "#F0E68C",
    "Mudstone": "#8B4513",
}

# Common log display settings
LOG_SETTINGS = {
    "GR": {"name": "Gamma Ray", "unit": "API", "color": "green", "range": [0, 150]},
    "RHOB": {
        "name": "Bulk Density",
        "unit": "g/cc",
        "color": "red",
        "range": [1.8, 2.8],
    },
    "NPHI": {
        "name": "Neutron Porosity",
        "unit": "v/v",
        "color": "blue",
        "range": [0.45, -0.15],  # Reversed scale
    },
    "RT": {
        "name": "Resistivity",
        "unit": "ohm.m",
        "color": "black",
        "range": [0.2, 2000],
        "log_scale": True,
    },
    "ILD": {
        "name": "Deep Resistivity",
        "unit": "ohm.m",
        "color": "black",
        "range": [0.2, 2000],
        "log_scale": True,
    },
    "PE": {
        "name": "Photo Electric",
        "unit": "b/e",
        "color": "purple",
        "range": [0, 10],
    },
    "DT": {
        "name": "Sonic",
        "unit": "us/ft",
        "color": "blue",
        "range": [140, 40],  # Reversed scale
    },
    "CALI": {"name": "Caliper", "unit": "in", "color": "orange", "range": [6, 16]},
}
