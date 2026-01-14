"""Well log plotting utilities.

Layer 4: Workflows - Public entry points with plotting.
"""

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from geosmith.workflows.plotting._base import (
    FACIES_COLORS,
    LOG_SETTINGS,
    MATPLOTLIB_AVAILABLE,
    Figure,
    _apply_plotsmith_style,
    _is_plotsmith_available,
)
from geosmith.workflows.plotting._base import plt  # noqa: F401

if TYPE_CHECKING:
    from matplotlib.axes import Axes

logger = logging.getLogger(__name__)


def create_strip_chart(
    df: pd.DataFrame,
    depth_col: str = "DEPTH",
    log_cols: Optional[List[str]] = None,
    facies_col: Optional[str] = None,
    title: str = "Well Log Strip Chart",
    figsize: Optional[Tuple[float, float]] = None,
    depth_range: Optional[Tuple[float, float]] = None,
    colors: Optional[Dict[str, str]] = None,
    use_plotsmith: Optional[bool] = None,
) -> Figure:
    """Create a strip chart (well log plot).

    Uses PlotSmith styling if available for cleaner, publication-ready plots.
    Falls back to matplotlib with clean styling when PlotSmith is not available.

    Args:
        df: DataFrame with well log data.
        depth_col: Name of depth column, default 'DEPTH'.
        log_cols: List of log column names to plot. If None, uses common logs.
        facies_col: Optional facies column name for facies track.
        title: Plot title, default 'Well Log Strip Chart'.
        figsize: Figure size (width, height) in inches. If None, auto-sized.
        depth_range: Optional (min, max) depth range to display.
        colors: Optional dictionary mapping log names to colors.
        use_plotsmith: If True, use PlotSmith styling (if available).
            If None, auto-detect.

    Returns:
        Matplotlib Figure object.

    Example:
        >>> from geosmith.workflows.plotting import create_strip_chart
        >>> import pandas as pd
        >>>
        >>> df = pd.DataFrame({
        ...     'DEPTH': np.linspace(0, 1000, 100),
        ...     'GR': np.random.normal(60, 15, 100),
        ...     'RHOB': np.random.normal(2.5, 0.2, 100)
        ... })
        >>> fig = create_strip_chart(df, log_cols=['GR', 'RHOB'])
        >>> fig.savefig('strip_chart.png')
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install geosmith[viz] or pip install matplotlib"
        )

    # Auto-detect PlotSmith if not specified
    if use_plotsmith is None:
        use_plotsmith = _is_plotsmith_available()

    if use_plotsmith:
        _apply_plotsmith_style()

    # Determine which logs to plot
    if log_cols is None:
        # Auto-detect available common logs
        common_logs = ["GR", "RHOB", "NPHI", "RT", "ILD", "PE", "DT", "CALI"]
        log_cols = [col for col in common_logs if col in df.columns]

        if not log_cols:
            # Fallback to any numeric columns except depth
            log_cols = [
                col
                for col in df.columns
                if col != depth_col and pd.api.types.is_numeric_dtype(df[col])
            ][:4]

    if not log_cols:
        raise ValueError("No log columns found to plot")

    # Apply depth range filter if specified
    if depth_range:
        df_plot = df[
            (df[depth_col] >= depth_range[0]) & (df[depth_col] <= depth_range[1])
        ].copy()
    else:
        df_plot = df.copy()

    # Add facies track if specified
    n_tracks = len(log_cols) + (1 if facies_col else 0)

    # Auto-size if not specified
    if figsize is None:
        width = min(2.5 * n_tracks, 15)
        height = 10
        figsize = (width, height)

    # Create subplots
    fig, axes = plt.subplots(1, n_tracks, figsize=figsize, sharey=True)

    # Handle single track case
    if n_tracks == 1:
        axes = [axes]

    # Plot each log track
    for idx, col in enumerate(log_cols):
        ax = axes[idx]
        add_log_track(ax, df_plot, depth_col, col, colors)

    # Add facies track if specified
    if facies_col and facies_col in df_plot.columns:
        ax = axes[-1]
        add_facies_track(ax, df_plot, depth_col, facies_col)

    # Set overall title
    fig.suptitle(title, fontsize=13, y=0.995)

    plt.tight_layout()
    return fig


def add_log_track(
    ax: "Axes",
    df: pd.DataFrame,
    depth_col: str,
    log_col: str,
    colors: Optional[Dict[str, str]] = None,
) -> None:
    """Add a single log track to an axis.

    Args:
        ax: Matplotlib axis.
        df: DataFrame with log data.
        depth_col: Name of depth column.
        log_col: Name of log column to plot.
        colors: Optional dictionary mapping log names to colors.
    """
    # Get settings for this log
    settings = LOG_SETTINGS.get(log_col, {})
    log_name = settings.get("name", log_col)
    unit = settings.get("unit", "")
    color = colors.get(log_col) if colors else settings.get("color", "black")
    log_range = settings.get("range")
    log_scale = settings.get("log_scale", False)

    # Plot the log
    depth = df[depth_col].values
    log_data = df[log_col].values

    # Filter out NaN values
    mask = ~(np.isnan(depth) | np.isnan(log_data))
    depth = depth[mask]
    log_data = log_data[mask]

    if len(depth) == 0:
        logger.warning(f"No valid data for {log_col}")
        return

    ax.plot(log_data, depth, color=color, linewidth=1)

    # Set axis properties
    if log_scale:
        ax.set_xscale("log")

    if log_range:
        ax.set_xlim(log_range)

    ax.set_xlabel(f"{log_name}\n({unit})" if unit else log_name, fontsize=10)

    # Invert y-axis for depth (geological convention)
    ax.invert_yaxis()

    # Clean styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle=":")


def add_facies_track(
    ax: "Axes",
    df: pd.DataFrame,
    depth_col: str,
    facies_col: str,
    facies_colors: Optional[Dict[str, str]] = None,
) -> None:
    """Add a facies track to an axis.

    Args:
        ax: Matplotlib axis.
        df: DataFrame with facies data.
        depth_col: Name of depth column.
        facies_col: Name of facies column.
        facies_colors: Optional dictionary mapping facies to colors.
    """
    if facies_colors is None:
        facies_colors = FACIES_COLORS

    depth = df[depth_col].values
    facies = df[facies_col].values

    # Get unique facies
    unique_facies = df[facies_col].dropna().unique()

    # Plot facies as colored bands
    for i in range(len(depth) - 1):
        facies_name = facies[i]
        if pd.notna(facies_name):
            color = facies_colors.get(str(facies_name), "#CCCCCC")
            ax.fill_betweenx([depth[i], depth[i + 1]], 0, 1, color=color, alpha=0.8)

    ax.set_xlabel("Facies", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.invert_yaxis()

    # Add legend
    handles = []
    labels = []
    for facies_name in sorted(unique_facies):
        if pd.notna(facies_name):
            facies_str = str(facies_name)
            color = facies_colors.get(facies_str, "#CCCCCC")
            handles.append(plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.8))
            labels.append(facies_str)

    if handles:
        ax.legend(
            handles,
            labels,
            loc="upper right",
            frameon=False,
            bbox_to_anchor=(1.0, 1.0),
            fontsize=8,
        )

    # Clean styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def create_facies_log_plot(
    df: pd.DataFrame,
    depth_col: str = "DEPTH",
    facies_col: str = "Facies",
    log_cols: Optional[List[str]] = None,
    title: str = "Well Logs with Facies",
    figsize: Optional[Tuple[float, float]] = None,
    depth_range: Optional[Tuple[float, float]] = None,
    use_plotsmith: Optional[bool] = None,
) -> Figure:
    """Create a strip chart with facies track.

    Convenience function that ensures facies are displayed.

    Args:
        df: DataFrame with well log and facies data.
        depth_col: Name of depth column, default 'DEPTH'.
        facies_col: Name of facies column, default 'Facies'.
        log_cols: List of log column names to plot.
        title: Plot title, default 'Well Logs with Facies'.
        figsize: Figure size (width, height) in inches.
        depth_range: Optional (min, max) depth range to display.
        use_plotsmith: If True, use PlotSmith styling (if available).
            If None, auto-detect.

    Returns:
        Matplotlib Figure object.
    """
    return create_strip_chart(
        df=df,
        depth_col=depth_col,
        log_cols=log_cols,
        facies_col=facies_col,
        title=title,
        figsize=figsize,
        depth_range=depth_range,
        use_plotsmith=use_plotsmith,
    )


def create_multi_well_strip_chart(
    dfs: List[pd.DataFrame],
    well_names: List[str],
    depth_col: str = "DEPTH",
    log_cols: Optional[List[str]] = None,
    facies_col: Optional[str] = None,
    title: str = "Multi-Well Strip Chart",
    figsize: Optional[Tuple[float, float]] = None,
    use_plotsmith: Optional[bool] = None,
) -> Figure:
    """Create strip charts for multiple wells side-by-side.

    Args:
        dfs: List of DataFrames, one per well.
        well_names: List of well names.
        depth_col: Name of depth column, default 'DEPTH'.
        log_cols: List of log column names to plot.
        facies_col: Optional facies column name.
        title: Plot title, default 'Multi-Well Strip Chart'.
        figsize: Figure size (width, height) in inches.
        use_plotsmith: If True, use PlotSmith styling (if available).
            If None, auto-detect.

    Returns:
        Matplotlib Figure object.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install geosmith[viz] or pip install matplotlib"
        )

    if len(dfs) != len(well_names):
        raise ValueError("Number of DataFrames must match number of well names")

    # Auto-detect PlotSmith if not specified
    if use_plotsmith is None:
        use_plotsmith = _is_plotsmith_available()

    if use_plotsmith:
        _apply_plotsmith_style()

    if log_cols is None:
        # Auto-detect from first well
        common_logs = ["GR", "RHOB", "NPHI", "RT"]
        log_cols = [col for col in common_logs if col in dfs[0].columns] if dfs else []

    if not log_cols:
        raise ValueError("No log columns found to plot")

    n_wells = len(dfs)
    n_tracks_per_well = len(log_cols) + (1 if facies_col else 0)

    # Auto-size if not specified
    if figsize is None:
        width = min(2.5 * n_tracks_per_well * n_wells, 20)
        height = 10
        figsize = (width, height)

    # Create figure with subplots
    fig = plt.figure(figsize=figsize)

    for well_idx, (df, well_name) in enumerate(zip(dfs, well_names)):
        # Create subplots for this well
        for track_idx in range(n_tracks_per_well):
            ax_idx = well_idx * n_tracks_per_well + track_idx + 1
            ax = fig.add_subplot(
                1,
                n_wells * n_tracks_per_well,
                ax_idx,
                sharey=(ax_idx > 1) if ax_idx > 1 else None,
            )

            if track_idx < len(log_cols):
                # Log track
                add_log_track(ax, df, depth_col, log_cols[track_idx], None)

                # Add well name to first track
                if track_idx == 0:
                    ax.text(
                        0.5,
                        1.02,
                        well_name,
                        transform=ax.transAxes,
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        fontweight="bold",
                    )
            elif facies_col and facies_col in df.columns:
                # Facies track
                add_facies_track(ax, df, depth_col, facies_col)

            # Only show y-label on leftmost plot
            if well_idx > 0:
                ax.set_ylabel("")

    fig.suptitle(title, fontsize=13, y=0.995)
    plt.tight_layout()

    return fig

