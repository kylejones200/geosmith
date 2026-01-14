"""Geomechanics plotting utilities.

Layer 4: Workflows - Public entry points with plotting.
"""

from typing import Optional, Tuple

import pandas as pd

from geosmith.workflows.plotting._base import MATPLOTLIB_AVAILABLE, Figure
from geosmith.workflows.plotting._base import plt  # noqa: F401


def plot_pressure_profile(
    df: pd.DataFrame,
    depth_col: str = "Depth",
    pressure_cols: Optional[list] = None,
    title: str = "Pressure Profile",
    depth_units: str = "m",
    pressure_units: str = "MPa",
    figsize: Tuple[float, float] = (6, 8),
) -> Figure:
    """Create pressure profile plot.

    Visualizes pressure profiles (overburden, hydrostatic, pore pressure, etc.)
    as a function of depth.

    Args:
        df: DataFrame with depth and pressure data.
        depth_col: Name of depth column, default 'Depth'.
        pressure_cols: List of pressure column names (e.g., ['Sv', 'Ph', 'Pp']).
                      If None, uses ['Sv', 'Ph', 'Pp'].
        title: Plot title, default 'Pressure Profile'.
        depth_units: Units for depth axis, default 'm'.
        pressure_units: Units for pressure axis, default 'MPa'.
        figsize: Figure size (width, height) in inches, default (6, 8).

    Returns:
        Matplotlib Figure object.

    Example:
        >>> from geosmith.workflows.plotting import plot_pressure_profile
        >>>
        >>> fig = plot_pressure_profile(
        ...     df, depth_col='DEPTH', pressure_cols=['Sv', 'Ph', 'Pp']
        ... )
        >>> fig.savefig('pressure_profile.png')

    Raises:
        ImportError: If matplotlib is not available.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install geosmith[viz] or pip install matplotlib"
        )

    if pressure_cols is None:
        pressure_cols = ["Sv", "Ph", "Pp"]

    fig, ax = plt.subplots(figsize=figsize)

    colors = {
        "Sv": "black",
        "Ph": "blue",
        "Pp": "red",
        "Shmin": "green",
        "SHmax": "orange",
        "Sigma_eff": "purple",
    }

    linestyles = {
        "Sv": "-",
        "Ph": "--",
        "Pp": "-",
        "Shmin": "-.",
        "SHmax": ":",
        "Sigma_eff": "--",
    }

    for col in pressure_cols:
        if col in df.columns:
            ax.plot(
                df[col],
                df[depth_col],
                color=colors.get(col, "gray"),
                linestyle=linestyles.get(col, "-"),
                linewidth=1.5,
                label=col,
            )

    # Labels and title
    ax.set_xlabel(f"Pressure ({pressure_units})")
    ax.set_ylabel(f"Depth ({depth_units})")
    ax.set_title(title)

    # Invert y-axis for depth (geological convention)
    ax.invert_yaxis()

    # Legend
    ax.legend(loc="best")

    # Clean up spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Grid
    ax.grid(True, alpha=0.3, linestyle=":")

    plt.tight_layout()
    return fig


def plot_mud_weight_profile(
    df: pd.DataFrame,
    depth_col: str = "Depth",
    mw_cols: Optional[list] = None,
    title: str = "Equivalent Mud Weight",
    depth_units: str = "m",
    figsize: Tuple[float, float] = (6, 8),
) -> Figure:
    """Create equivalent mud weight profile plot.

    Visualizes mud weight profiles as a function of depth, useful for well planning.

    Args:
        df: DataFrame with depth and mud weight data.
        depth_col: Name of depth column, default 'Depth'.
        mw_cols: List of mud weight column names. If None, auto-detects columns
                containing 'MW' or 'mud_weight'.
        title: Plot title, default 'Equivalent Mud Weight'.
        depth_units: Units for depth axis, default 'm'.
        figsize: Figure size (width, height) in inches, default (6, 8).

    Returns:
        Matplotlib Figure object.

    Example:
        >>> from geosmith.workflows.plotting import plot_mud_weight_profile
        >>>
        >>> fig = plot_mud_weight_profile(
        ...     df, depth_col='DEPTH', mw_cols=['MW_min', 'MW_max']
        ... )
        >>> fig.savefig('mud_weight_profile.png')

    Raises:
        ImportError: If matplotlib is not available.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install geosmith[viz] or pip install matplotlib"
        )

    if mw_cols is None:
        mw_cols = [
            col for col in df.columns if "MW" in col or "mud_weight" in col.lower()
        ]

    fig, ax = plt.subplots(figsize=figsize)

    for col in mw_cols:
        if col in df.columns:
            ax.plot(df[col], df[depth_col], linewidth=1.5, label=col)

    # Labels and title
    ax.set_xlabel("Mud Weight (g/cc)", fontsize=11)
    ax.set_ylabel(f"Depth ({depth_units})", fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)

    # Invert y-axis for depth
    ax.invert_yaxis()

    # Legend
    ax.legend(frameon=False, fontsize=9, loc="best")

    # Clean up spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Grid
    ax.grid(True, alpha=0.3, linestyle=":")

    plt.tight_layout()
    return fig

