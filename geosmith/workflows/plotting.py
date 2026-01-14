"""Petrophysics plotting utilities.

Layer 4: Workflows - Public entry points with plotting.
Migrated from geosuite.petro.buckles, geosuite.petro.pickett, geosuite.petro.lithology.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

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


def buckles_plot(
    df: pd.DataFrame,
    porosity_col: str = "PHIND",
    sw_col: str = "SW",
    cutoff: float = 0.04,
    title: str = "Buckles Plot",
    color_by: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
) -> Figure:
    """Create a Buckles plot for reservoir quality analysis.

    The Buckles plot (Bulk Volume Water vs Porosity) is used to identify
    productive reservoir zones. The BVW cutoff separates water-bearing
    from hydrocarbon-bearing zones.

    Args:
        df: DataFrame with petrophysical data.
        porosity_col: Column name for porosity, default 'PHIND'.
        sw_col: Column name for water saturation, default 'SW'.
        cutoff: BVW cutoff value (typically 0.03-0.05), default 0.04.
        title: Plot title, default 'Buckles Plot'.
        color_by: Optional column to color points by.
        figsize: Figure size (width, height) in inches, default (8, 6).

    Returns:
        Matplotlib Figure object.

    Example:
        >>> from geosmith.workflows.plotting import buckles_plot
        >>>
        >>> fig = buckles_plot(df, porosity_col='PHIND', sw_col='SW', cutoff=0.04)
        >>> fig.savefig('buckles.png')

    Raises:
        ImportError: If matplotlib is not available.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install geosmith[viz] or pip install matplotlib"
        )

    # Calculate Bulk Volume Water
    df_plot = df[[porosity_col, sw_col]].dropna()
    df_plot = df_plot[(df_plot[porosity_col] > 0) & (df_plot[sw_col] > 0)]

    phi = df_plot[porosity_col].values
    sw = df_plot[sw_col].values
    bvw = phi * sw

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot data points
    if color_by and color_by in df.columns:
        color_data = df.loc[df_plot.index, color_by]
        scatter = ax.scatter(phi, bvw, c=color_data, s=30, alpha=0.6, cmap="viridis")
        plt.colorbar(scatter, ax=ax, label=color_by)
    else:
        scatter = ax.scatter(phi, bvw, c=bvw, s=30, alpha=0.6, cmap="RdYlGn_r")
        plt.colorbar(scatter, ax=ax, label="BVW")

    # Add cutoff line
    phi_range = np.linspace(0, phi.max() * 1.1, 100)
    ax.axhline(
        y=cutoff,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"BVW Cutoff = {cutoff}",
    )

    # Add Sw contours
    for sw_line in [0.2, 0.4, 0.6, 0.8, 1.0]:
        bvw_line = phi_range * sw_line
        ax.plot(phi_range, bvw_line, "k:", linewidth=0.8, alpha=0.5)
        # Add label at the end of the line
        if sw_line in [0.2, 0.6, 1.0]:
            ax.text(
                phi.max() * 1.05,
                phi.max() * sw_line * 1.05,
                f"Sw={sw_line}",
                fontsize=8,
                alpha=0.7,
            )

    # Labels and title
    ax.set_xlabel("Porosity (v/v)", fontsize=11)
    ax.set_ylabel("Bulk Volume Water (BVW)", fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)

    # Set axis limits
    ax.set_xlim(0, phi.max() * 1.1)
    ax.set_ylim(0, max(bvw.max() * 1.1, cutoff * 2))

    # Add zone labels
    ax.text(
        phi.max() * 0.5,
        cutoff * 0.5,
        "Hydrocarbon Zone",
        ha="center",
        va="center",
        fontsize=10,
        color="green",
        alpha=0.7,
    )
    ax.text(
        phi.max() * 0.5,
        cutoff * 1.5,
        "Water Zone",
        ha="center",
        va="center",
        fontsize=10,
        color="red",
        alpha=0.7,
    )

    # Legend
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    # Clean up spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig


def pickett_plot(
    df: pd.DataFrame,
    porosity_col: str = "NPHI",
    resistivity_col: str = "RT",
    m: float = 2.0,
    n: float = 2.0,
    a: float = 1.0,
    rw: float = 0.05,
    color_by: Optional[str] = None,
    title: str = "Pickett Plot",
    show_water_line: bool = True,
    show_sw_lines: bool = True,
    figsize: Tuple[float, float] = (8, 6),
) -> Figure:
    """Create a Pickett plot for porosity and resistivity analysis.

    The Pickett plot is a log-log crossplot of porosity vs resistivity
    used to determine water saturation and identify reservoir quality.

    Args:
        df: DataFrame with well log data.
        porosity_col: Column name for porosity (fraction, not %), default 'NPHI'.
        resistivity_col: Column name for resistivity, default 'RT'.
        m: Cementation exponent (typically 1.8-2.5), default 2.0.
        n: Saturation exponent (typically ~2), default 2.0.
        a: Tortuosity factor (typically 0.6-1.4), default 1.0.
        rw: Formation water resistivity (ohm-m), default 0.05.
        color_by: Optional column to color points by.
        title: Plot title, default 'Pickett Plot'.
        show_water_line: If True, show 100% water saturation line, default True.
        show_sw_lines: If True, show Sw = 50%, 25% lines, default True.
        figsize: Figure size (width, height) in inches, default (8, 6).

    Returns:
        Matplotlib Figure object.

    Example:
        >>> from geosmith.workflows.plotting import pickett_plot
        >>>
        >>> fig = pickett_plot(df, porosity_col='NPHI', resistivity_col='RT', m=2.0)
        >>> fig.savefig('pickett.png')

    Raises:
        ImportError: If matplotlib is not available.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install geosmith[viz] or pip install matplotlib"
        )

    # Remove invalid data
    df_plot = df[[porosity_col, resistivity_col]].dropna()
    df_plot = df_plot[(df_plot[porosity_col] > 0) & (df_plot[resistivity_col] > 0)]

    phi = df_plot[porosity_col].values
    rt = df_plot[resistivity_col].values

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot data points
    if color_by and color_by in df.columns:
        color_data = df.loc[df_plot.index, color_by]
        scatter = ax.scatter(phi, rt, c=color_data, s=20, alpha=0.6, cmap="viridis")
        plt.colorbar(scatter, ax=ax, label=color_by)
    else:
        ax.scatter(phi, rt, s=20, alpha=0.6, color="black")

    # Add water saturation lines
    if show_water_line or show_sw_lines:
        phi_range = np.logspace(np.log10(phi.min()), np.log10(phi.max()), 100)

        if show_water_line:
            rt_100 = (a * rw) / (phi_range**m)
            ax.plot(phi_range, rt_100, "k--", linewidth=1.5, label="Sw = 100%")

        if show_sw_lines:
            rt_50 = (a * rw) / ((0.5**n) * (phi_range**m))
            ax.plot(phi_range, rt_50, "k:", linewidth=1, label="Sw = 50%")

            rt_25 = (a * rw) / ((0.25**n) * (phi_range**m))
            ax.plot(phi_range, rt_25, "k:", linewidth=1, label="Sw = 25%", alpha=0.7)

    # Set log scales
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Labels and title
    ax.set_xlabel("Porosity (v/v)")
    ax.set_ylabel("Resistivity (ohm-m)")
    ax.set_title(title)

    # Legend
    if show_water_line or show_sw_lines:
        ax.legend(loc="best")

    # Add parameter text
    param_text = f"m={m}, n={n}, a={a}, Rw={rw}"
    ax.text(
        0.02,
        0.98,
        param_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    # Clean up spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig


def neutron_density_crossplot(
    df: pd.DataFrame,
    nphi_col: str = "NPHI",
    rhob_col: str = "RHOB",
    color_by: Optional[str] = None,
    title: str = "Neutron-Density Crossplot",
    show_lithology_lines: bool = True,
    figsize: Tuple[float, float] = (8, 6),
) -> Figure:
    """Create neutron-density crossplot for lithology identification.

    This plot helps identify different rock types and fluid content
    based on their neutron and density log responses.

    Args:
        df: DataFrame with log data.
        nphi_col: Neutron porosity column, default 'NPHI'.
        rhob_col: Bulk density column, default 'RHOB'.
        color_by: Optional column to color points by.
        title: Plot title, default 'Neutron-Density Crossplot'.
        show_lithology_lines: If True, show lithology reference lines, default True.
        figsize: Figure size (width, height) in inches, default (8, 6).

    Returns:
        Matplotlib Figure object.

    Example:
        >>> from geosmith.workflows.plotting import neutron_density_crossplot
        >>>
        >>> fig = neutron_density_crossplot(df, nphi_col='NPHI', rhob_col='RHOB')
        >>> fig.savefig('neutron_density.png')

    Raises:
        ImportError: If matplotlib is not available.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install geosmith[viz] or pip install matplotlib"
        )

    # Remove invalid data
    df_plot = df[[nphi_col, rhob_col]].dropna()

    nphi = df_plot[nphi_col].values
    rhob = df_plot[rhob_col].values

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Add lithology reference lines first (behind data)
    if show_lithology_lines:
        # Sandstone line (typical values)
        ss_phi = np.linspace(0, 0.4, 50)
        ss_rhob = 2.65 - ss_phi * (2.65 - 1.0)  # sandstone matrix = 2.65
        ax.plot(
            ss_phi,
            ss_rhob,
            "-",
            color="gold",
            linewidth=2,
            label="Sandstone",
            alpha=0.7,
        )

        # Limestone line
        ls_phi = np.linspace(0, 0.4, 50)
        ls_rhob = 2.71 - ls_phi * (2.71 - 1.0)  # limestone matrix = 2.71
        ax.plot(
            ls_phi,
            ls_rhob,
            "-",
            color="gray",
            linewidth=2,
            label="Limestone",
            alpha=0.7,
        )

        # Dolomite line
        dol_phi = np.linspace(0, 0.4, 50)
        dol_rhob = 2.87 - dol_phi * (2.87 - 1.0)  # dolomite matrix = 2.87
        ax.plot(
            dol_phi,
            dol_rhob,
            "-",
            color="saddlebrown",
            linewidth=2,
            label="Dolomite",
            alpha=0.7,
        )

    # Plot data points
    if color_by and color_by in df.columns:
        color_data = df.loc[df_plot.index, color_by]
        scatter = ax.scatter(
            nphi, rhob, c=color_data, s=20, alpha=0.6, cmap="viridis", zorder=3
        )
        plt.colorbar(scatter, ax=ax, label=color_by)
    else:
        ax.scatter(nphi, rhob, s=20, alpha=0.6, color="black", zorder=3)

    # Labels and title
    ax.set_xlabel("Neutron Porosity (v/v)", fontsize=11)
    ax.set_ylabel("Bulk Density (g/cc)", fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)

    # Set axis limits and invert y-axis
    ax.set_xlim(-0.05, 0.45)
    ax.set_ylim(3.0, 1.8)  # Inverted

    # Legend
    if show_lithology_lines:
        ax.legend(frameon=False, fontsize=9, loc="upper right")

    # Clean up spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig


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


# =============================================================================
# Well Log Strip Charts
# =============================================================================


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

            # PlotSmith may have a style.apply() or similar
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


# =============================================================================
# Ternary Plots
# =============================================================================


def _ternary_to_cartesian(
    a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert ternary coordinates (a, b, c) to Cartesian (x, y).

    Ternary coordinates must sum to 1 (or 100 if using percentages).
    The triangle vertices are:
    - Top vertex: (0, 1) for component 'a'
    - Bottom-left: (0, 0) for component 'b'
    - Bottom-right: (1, 0) for component 'c'

    Args:
        a: First component (0-1 or 0-100).
        b: Second component (0-1 or 0-100).
        c: Third component (0-1 or 0-100).

    Returns:
        Tuple of (x, y) arrays in Cartesian coordinates.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)

    # Normalize if sum > 1.5 (assuming percentages)
    total = a + b + c
    if np.any(total > 1.5):
        a = a / total
        b = b / total
        c = c / total

    # Convert to Cartesian coordinates
    # x = (b + 2*c) / 2
    # y = (sqrt(3) * b) / 2
    x = (b + 2 * c) / 2.0
    y = (np.sqrt(3) * b) / 2.0

    return x, y


def _cartesian_to_ternary(
    x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert Cartesian coordinates (x, y) to ternary (a, b, c).

    Args:
        x: X coordinates (0-1).
        y: Y coordinates (0-1).

    Returns:
        Tuple of (a, b, c) arrays in ternary coordinates.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Convert from Cartesian to ternary
    b = (2 * y) / np.sqrt(3)
    c = x - (b / 2)
    a = 1 - b - c

    return a, b, c


def _setup_ternary_axes(
    ax: Axes,
    labels: Tuple[str, str, str] = ("A", "B", "C"),
    grid: bool = True,
    grid_style: str = "dashed",
    grid_alpha: float = 0.3,
) -> None:
    """Set up ternary plot axes with triangle frame and grid lines.

    Args:
        ax: Matplotlib axes.
        labels: Tuple of (top, bottom-left, bottom-right) labels.
        grid: If True, draw grid lines, default True.
        grid_style: Grid line style ('dashed', 'dotted', 'solid'), default 'dashed'.
        grid_alpha: Grid line alpha (transparency), default 0.3.
    """
    if not MATPLOTLIB_AVAILABLE or mpatches is None:
        raise ImportError("matplotlib is required for ternary plots")

    # Clear axis
    ax.clear()

    # Draw triangle frame
    triangle_vertices = np.array(
        [
            [0.5, np.sqrt(3) / 2],  # Top vertex
            [0, 0],  # Bottom-left
            [1, 0],  # Bottom-right
        ]
    )

    triangle = mpatches.Polygon(
        triangle_vertices, closed=True, fill=False, edgecolor="black", linewidth=1.5
    )
    ax.add_patch(triangle)

    # Add grid lines if requested
    if grid:
        # Grid lines parallel to each side (10 divisions)
        n_grid = 10

        # Lines parallel to bottom (horizontal)
        for i in range(1, n_grid):
            y_val = i * np.sqrt(3) / (2 * n_grid)
            x_start = (1 - (2 * i / n_grid)) / 2
            x_end = 1 - x_start
            ax.plot(
                [x_start, x_end],
                [y_val, y_val],
                linestyle=grid_style,
                color="gray",
                alpha=grid_alpha,
                linewidth=0.5,
            )

        # Lines parallel to right side (slanted left)
        for i in range(1, n_grid):
            y_start = 0
            x_start = i / n_grid
            y_end = np.sqrt(3) * (1 - i / n_grid) / 2
            x_end = 0.5 + i / (2 * n_grid)
            ax.plot(
                [x_start, x_end],
                [y_start, y_end],
                linestyle=grid_style,
                color="gray",
                alpha=grid_alpha,
                linewidth=0.5,
            )

        # Lines parallel to left side (slanted right)
        for i in range(1, n_grid):
            y_start = 0
            x_start = 1 - i / n_grid
            y_end = np.sqrt(3) * (1 - i / n_grid) / 2
            x_end = 0.5 - i / (2 * n_grid)
            ax.plot(
                [x_start, x_end],
                [y_start, y_end],
                linestyle=grid_style,
                color="gray",
                alpha=grid_alpha,
                linewidth=0.5,
            )

    # Add axis labels
    ax.text(
        0.5,
        np.sqrt(3) / 2 + 0.05,
        labels[0],
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )
    ax.text(
        -0.05, -0.05, labels[1], ha="right", va="top", fontsize=11, fontweight="bold"
    )
    ax.text(1.05, -0.05, labels[2], ha="left", va="top", fontsize=11, fontweight="bold")

    # Set axis limits with padding
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, np.sqrt(3) / 2 + 0.1)
    ax.set_aspect("equal")
    ax.axis("off")


def ternary_plot(
    df: pd.DataFrame,
    a_col: str,
    b_col: str,
    c_col: str,
    labels: Optional[Tuple[str, str, str]] = None,
    color_by: Optional[str] = None,
    size_by: Optional[str] = None,
    title: str = "Ternary Plot",
    figsize: Tuple[float, float] = (8, 8),
    grid: bool = True,
    cmap: str = "viridis",
    alpha: float = 0.7,
    s: float = 30,
    use_plotsmith: Optional[bool] = None,
) -> Figure:
    """Create a ternary plot from three-component data.

    Uses PlotSmith styling if available for cleaner, publication-ready plots.
    Falls back to matplotlib with clean styling when PlotSmith is not available.

    Args:
        df: DataFrame with three-component data.
        a_col: Column name for first component (top vertex).
        b_col: Column name for second component (bottom-left).
        c_col: Column name for third component (bottom-right).
        labels: Tuple of (a_label, b_label, c_label). If None, uses column names.
        color_by: Optional column to color points by.
        size_by: Optional column to size points by.
        title: Plot title, default 'Ternary Plot'.
        figsize: Figure size (width, height) in inches, default (8, 8).
        grid: If True, show grid lines, default True.
        cmap: Colormap for coloring, default 'viridis'.
        alpha: Point transparency (0-1), default 0.7.
        s: Point size (if size_by not used), default 30.
        use_plotsmith: If True, use PlotSmith styling (if available).
            If None, auto-detect.

    Returns:
        Matplotlib Figure object.

    Example:
        >>> from geosmith.workflows.plotting import ternary_plot
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> df = pd.DataFrame({
        ...     'Sand': [0.5, 0.3, 0.7],
        ...     'Silt': [0.3, 0.5, 0.2],
        ...     'Clay': [0.2, 0.2, 0.1]
        ... })
        >>> fig = ternary_plot(
        ...     df, 'Sand', 'Silt', 'Clay',
        ...     labels=('Sand', 'Silt', 'Clay'),
        ...     title='Texture Classification'
        ... )
        >>> fig.savefig('ternary.png')
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

    if labels is None:
        labels = (a_col, b_col, c_col)

    # Extract components
    a = df[a_col].values
    b = df[b_col].values
    c = df[c_col].values

    # Remove invalid data
    mask = ~(np.isnan(a) | np.isnan(b) | np.isnan(c))
    a = a[mask]
    b = b[mask]
    c = c[mask]

    if len(a) == 0:
        raise ValueError("No valid data points after removing NaNs")

    # Convert to Cartesian coordinates
    x, y = _ternary_to_cartesian(a, b, c)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Set up ternary axes
    _setup_ternary_axes(ax, labels=labels, grid=grid)

    # Prepare point properties
    colors = None
    sizes = np.full(len(x), s)

    if color_by and color_by in df.columns:
        color_data = df[color_by].values[mask]
        colors = color_data

    if size_by and size_by in df.columns:
        size_data = df[size_by].values[mask]
        # Normalize sizes to reasonable range
        size_min, size_max = size_data.min(), size_data.max()
        if size_max > size_min:
            sizes = s * (0.5 + 1.5 * (size_data - size_min) / (size_max - size_min))
        else:
            sizes = np.full(len(x), s)

    # Plot points
    if colors is not None:
        scatter = ax.scatter(
            x,
            y,
            c=colors,
            s=sizes,
            alpha=alpha,
            cmap=cmap,
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )
        cbar = plt.colorbar(scatter, ax=ax, pad=0.05, shrink=0.8)
        cbar.set_label(color_by, fontsize=10)
    else:
        ax.scatter(
            x,
            y,
            s=sizes,
            alpha=alpha,
            color="black",
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )

    # Set title
    ax.set_title(title, fontsize=13, pad=20, fontweight="bold")

    plt.tight_layout()
    return fig


def sand_silt_clay_plot(
    df: pd.DataFrame,
    sand_col: str = "Sand",
    silt_col: str = "Silt",
    clay_col: str = "Clay",
    color_by: Optional[str] = None,
    title: str = "Sand-Silt-Clay Texture Classification",
    show_classification: bool = True,
    figsize: Tuple[float, float] = (10, 10),
    use_plotsmith: Optional[bool] = None,
    **kwargs: Any,
) -> Figure:
    """Create a sand-silt-clay ternary plot for soil/sediment texture classification.

    This is a specialized ternary plot with USDA texture classification zones.

    Args:
        df: DataFrame with sand, silt, clay percentages.
        sand_col: Column name for sand percentage, default 'Sand'.
        silt_col: Column name for silt percentage, default 'Silt'.
        clay_col: Column name for clay percentage, default 'Clay'.
        color_by: Optional column to color points by.
        title: Plot title, default 'Sand-Silt-Clay Texture Classification'.
        show_classification: If True, draw texture classification boundaries,
            default True.
        figsize: Figure size (width, height) in inches, default (10, 10).
        use_plotsmith: If True, use PlotSmith styling (if available).
            If None, auto-detect.
        **kwargs: Additional arguments passed to ternary_plot.

    Returns:
        Matplotlib Figure object.

    Example:
        >>> from geosmith.workflows.plotting import sand_silt_clay_plot
        >>> import pandas as pd
        >>>
        >>> df = pd.DataFrame({
        ...     'Sand': [50, 30, 70],
        ...     'Silt': [30, 50, 20],
        ...     'Clay': [20, 20, 10]
        ... })
        >>> fig = sand_silt_clay_plot(df, title='Sediment Texture')
        >>> fig.savefig('texture.png')
    """
    fig = ternary_plot(
        df=df,
        a_col=sand_col,
        b_col=silt_col,
        c_col=clay_col,
        labels=("Sand", "Silt", "Clay"),
        title=title,
        figsize=figsize,
        color_by=color_by,
        use_plotsmith=use_plotsmith,
        **kwargs,
    )

    # Add texture classification zones if requested
    if show_classification:
        # Note: Full USDA texture classification requires defining polygon vertices
        # for each texture class (e.g., Loam, Sandy Loam, Clay Loam, etc.)
        # This is a placeholder for future enhancement
        # Common texture classes boundaries can be added as polygon patches
        logger.debug(
            "Texture classification zones can be enhanced by adding polygon patches"
        )

    return fig


def qfl_plot(
    df: pd.DataFrame,
    q_col: str = "Quartz",
    f_col: str = "Feldspar",
    l_col: str = "Lithics",
    color_by: Optional[str] = None,
    title: str = "Q-F-L Rock Classification",
    figsize: Tuple[float, float] = (10, 10),
    use_plotsmith: Optional[bool] = None,
    **kwargs: Any,
) -> Figure:
    """Create a Q-F-L (Quartz-Feldspar-Lithics) ternary plot for rock classification.

    This ternary diagram is commonly used in sedimentary petrology to classify
    sandstones and other clastic rocks.

    Args:
        df: DataFrame with Q, F, L percentages.
        q_col: Column name for Quartz percentage, default 'Quartz'.
        f_col: Column name for Feldspar percentage, default 'Feldspar'.
        l_col: Column name for Lithics percentage, default 'Lithics'.
        color_by: Optional column to color points by.
        title: Plot title, default 'Q-F-L Rock Classification'.
        figsize: Figure size (width, height) in inches, default (10, 10).
        use_plotsmith: If True, use PlotSmith styling (if available).
            If None, auto-detect.
        **kwargs: Additional arguments passed to ternary_plot.

    Returns:
        Matplotlib Figure object.

    Example:
        >>> from geosmith.workflows.plotting import qfl_plot
        >>> import pandas as pd
        >>>
        >>> df = pd.DataFrame({
        ...     'Quartz': [60, 40, 80],
        ...     'Feldspar': [25, 40, 10],
        ...     'Lithics': [15, 20, 10]
        ... })
        >>> fig = qfl_plot(df, title='Sandstone Classification')
        >>> fig.savefig('qfl.png')
    """
    return ternary_plot(
        df=df,
        a_col=q_col,
        b_col=f_col,
        c_col=l_col,
        labels=("Quartz", "Feldspar", "Lithics"),
        title=title,
        figsize=figsize,
        color_by=color_by,
        use_plotsmith=use_plotsmith,
        **kwargs,
    )


def mineral_composition_plot(
    df: pd.DataFrame,
    mineral1_col: str,
    mineral2_col: str,
    mineral3_col: str,
    labels: Optional[Tuple[str, str, str]] = None,
    color_by: Optional[str] = None,
    title: str = "Mineral Composition",
    figsize: Tuple[float, float] = (10, 10),
    use_plotsmith: Optional[bool] = None,
    **kwargs: Any,
) -> Figure:
    """Create a ternary plot for mineral composition analysis.

    General-purpose ternary plot for any three-component mineral composition.

    Args:
        df: DataFrame with mineral percentages.
        mineral1_col: Column name for first mineral.
        mineral2_col: Column name for second mineral.
        mineral3_col: Column name for third mineral.
        labels: Tuple of mineral labels. If None, uses column names.
        color_by: Optional column to color points by.
        title: Plot title, default 'Mineral Composition'.
        figsize: Figure size (width, height) in inches, default (10, 10).
        use_plotsmith: If True, use PlotSmith styling (if available).
            If None, auto-detect.
        **kwargs: Additional arguments passed to ternary_plot.

    Returns:
        Matplotlib Figure object.

    Example:
        >>> from geosmith.workflows.plotting import mineral_composition_plot
        >>> import pandas as pd
        >>>
        >>> df = pd.DataFrame({
        ...     'Quartz': [40, 50, 60],
        ...     'Feldspar': [30, 25, 20],
        ...     'Mica': [30, 25, 20]
        ... })
        >>> fig = mineral_composition_plot(
        ...     df, 'Quartz', 'Feldspar', 'Mica',
        ...     title='Rock Mineralogy'
        ... )
        >>> fig.savefig('mineral.png')
    """
    if labels is None:
        labels = (mineral1_col, mineral2_col, mineral3_col)

    return ternary_plot(
        df=df,
        a_col=mineral1_col,
        b_col=mineral2_col,
        c_col=mineral3_col,
        labels=labels,
        title=title,
        figsize=figsize,
        color_by=color_by,
        use_plotsmith=use_plotsmith,
        **kwargs,
    )


# =============================================================================
# Interactive Maps (Folium)
# =============================================================================

# Optional Folium dependency
try:
    import folium
    from folium import plugins
    from folium.plugins import HeatMap

    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    folium = None  # type: ignore
    plugins = None  # type: ignore
    HeatMap = None  # type: ignore
    logger.debug(
        "folium not available. Interactive maps require folium. "
        "Install with: pip install folium"
    )


def create_interactive_kriging_map(
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    interpolated: np.ndarray,
    coordinates: Optional[np.ndarray] = None,
    values: Optional[np.ndarray] = None,
    variance: Optional[np.ndarray] = None,
    output_file: Optional[Union[str, Path]] = None,
    title: str = "Kriging Interpolation Map",
    production_type: str = "Production",
    sample_step: int = 5,
    max_wells: int = 1000,
    zoom_start: int = 8,
) -> Optional[Any]:
    """Create interactive Folium map from kriging results.

    Uses Folium for web-based interactive maps. PlotSmith styling is not applicable
    for Folium maps as they use HTML/JavaScript.

    Args:
        grid_lon: Grid longitude coordinates (1D array).
        grid_lat: Grid latitude coordinates (1D array).
        interpolated: Interpolated values (2D array, shape: [lat, lon]).
        coordinates: Optional well coordinates (n_wells, 2) - [lon, lat].
        values: Optional well production values (n_wells,).
        variance: Optional kriging variance (2D array, same shape as interpolated).
        output_file: Optional output file path for saving HTML.
        title: Map title, default 'Kriging Interpolation Map'.
        production_type: Production type label, default 'Production'.
        sample_step: Step size for sampling grid points (for performance), default 5.
        max_wells: Maximum number of wells to display, default 1000.
        zoom_start: Initial zoom level, default 8.

    Returns:
        Folium map object, or None if folium not available.

    Example:
        >>> from geosmith.workflows.plotting import create_interactive_kriging_map
        >>> import numpy as np
        >>>
        >>> grid_lon = np.linspace(-120, -118, 100)
        >>> grid_lat = np.linspace(38, 40, 100)
        >>> interpolated = np.random.rand(100, 100) * 1000
        >>> m = create_interactive_kriging_map(grid_lon, grid_lat, interpolated)
        >>> if m:
        ...     m.save('kriging_map.html')
    """
    if not FOLIUM_AVAILABLE:
        logger.warning("folium not available, cannot create interactive map")
        return None

    grid_lon = np.asarray(grid_lon)
    grid_lat = np.asarray(grid_lat)
    interpolated = np.asarray(interpolated)

    # Calculate center of map
    center_lat = float(np.mean([grid_lat.min(), grid_lat.max()]))
    center_lon = float(np.mean([grid_lon.min(), grid_lon.max()]))

    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles="OpenStreetMap",
    )

    # Add satellite imagery as alternative tile layer
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
        overlay=False,
        control=True,
    ).add_to(m)

    grid_lon_sampled = grid_lon[::sample_step]
    grid_lat_sampled = grid_lat[::sample_step]

    # Handle 2D interpolated array
    if interpolated.ndim == 2:
        interpolated_sampled = interpolated[::sample_step, ::sample_step]
    else:
        logger.warning("Interpolated array must be 2D")
        interpolated_sampled = np.array([[0]])

    # Create contour-like visualization using circles
    valid_values = interpolated_sampled[~np.isnan(interpolated_sampled)]
    if len(valid_values) > 0 and MATPLOTLIB_AVAILABLE:
        vmin = float(valid_values.min())
        vmax = float(valid_values.max())

        # Create color scale (viridis-like)
        from matplotlib import cm, colors as mcolors

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.get_cmap("viridis")

        # Add interpolation surface as circles
        for i, lat in enumerate(grid_lat_sampled):
            for j, lon in enumerate(grid_lon_sampled):
                if (
                    i < interpolated_sampled.shape[0]
                    and j < interpolated_sampled.shape[1]
                ):
                    val = interpolated_sampled[i, j]
                    if not np.isnan(val):
                        color = mcolors.rgb2hex(cmap(norm(val))[:3])
                        folium.CircleMarker(
                            location=[float(lat), float(lon)],
                            radius=3,
                            popup=f"Interpolated: {val:.2f}",
                            tooltip=f"Value: {val:.2f}",
                            color=color,
                            fillColor=color,
                            fillOpacity=0.6,
                            weight=0,
                        ).add_to(m)

    # Add well locations if provided
    if coordinates is not None and values is not None:
        coordinates = np.asarray(coordinates)
        values = np.asarray(values)

        n_wells = min(max_wells, len(coordinates))
        if n_wells < len(coordinates):
            indices = np.random.choice(len(coordinates), n_wells, replace=False)
        else:
            indices = np.arange(len(coordinates))

        valid_values_wells = values[~np.isnan(values)]
        if len(valid_values_wells) > 0:
            vmin_wells = float(valid_values_wells.min())
            vmax_wells = float(valid_values_wells.max())
        else:
            vmin_wells = 0.0
            vmax_wells = 1.0

        well_data = []
        for idx in indices:
            if coordinates[idx].shape[0] >= 2:
                lon, lat = float(coordinates[idx, 0]), float(coordinates[idx, 1])
                val = float(values[idx]) if not np.isnan(values[idx]) else 0.0

                if not np.isnan(lon) and not np.isnan(lat):
                    # Color code by production value
                    val_norm = (val - vmin_wells) / (vmax_wells - vmin_wells + 1e-10)

                    # Create color gradient
                    if val_norm < 0.33:
                        color = "green"
                    elif val_norm < 0.67:
                        color = "orange"
                    else:
                        color = "red"

                    well_data.append([lat, lon, val])

                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=4,
                        popup=f"Well Production: {val:.2f}",
                        tooltip=f"Production: {val:.2f}",
                        color="black",
                        fillColor=color,
                        fillOpacity=0.7,
                        weight=1,
                    ).add_to(m)

        # Add heatmap layer
        if len(well_data) > 0 and HeatMap is not None:
            HeatMap(well_data, radius=15, blur=10, max_zoom=1).add_to(m)

        n_wells_displayed = len(well_data)
    else:
        n_wells_displayed = 0

    # Add legend
    legend_html = f"""
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 200px; height: 180px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <h4>{title}</h4>
    <p><b>Production Type:</b> {production_type}</p>
    <p><b>Wells Shown:</b> {n_wells_displayed:,}</p>
    <p><b>Interpolation:</b> Kriging</p>
    <p style="font-size:10px">Green = Low<br>
    Orange = Medium<br>
    Red = High</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Add layer control
    folium.LayerControl().add_to(m)

    # Save map if output file specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(output_path))
        logger.info(f"Interactive map saved to: {output_path}")

    return m


def create_interactive_well_map(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    value_col: Optional[str] = None,
    well_id_col: Optional[str] = None,
    output_file: Optional[Union[str, Path]] = None,
    title: str = "Well Locations Map",
    zoom_start: int = 8,
    max_wells: int = 1000,
) -> Optional[Any]:
    """Create interactive map from well location data.

    Uses Folium for web-based interactive maps.

    Args:
        df: DataFrame with well location data.
        lat_col: Column name for latitude, default 'latitude'.
        lon_col: Column name for longitude, default 'longitude'.
        value_col: Optional column name for production values.
        well_id_col: Optional column name for well identifiers.
        output_file: Optional output file path.
        title: Map title, default 'Well Locations Map'.
        zoom_start: Initial zoom level, default 8.
        max_wells: Maximum number of wells to display, default 1000.

    Returns:
        Folium map object, or None if folium not available.

    Example:
        >>> from geosmith.workflows.plotting import create_interactive_well_map
        >>> import pandas as pd
        >>>
        >>> df = pd.DataFrame({
        ...     'latitude': [38.5, 39.0, 39.5],
        ...     'longitude': [-120.0, -119.5, -119.0],
        ...     'production': [1000, 2000, 1500]
        ... })
        >>> m = create_interactive_well_map(df, value_col='production')
        >>> if m:
        ...     m.save('wells_map.html')
    """
    if not FOLIUM_AVAILABLE:
        logger.warning("folium not available, cannot create interactive map")
        return None

    # Filter valid data
    df_clean = df[df[lat_col].notna() & df[lon_col].notna()].copy()

    if len(df_clean) == 0:
        logger.warning("No valid well locations found")
        return None

    # Sample if too many wells
    if len(df_clean) > max_wells:
        df_clean = df_clean.sample(n=max_wells, random_state=42)
        logger.info(f"Sampled {max_wells} wells for display")

    # Calculate center
    center_lat = float(df_clean[lat_col].mean())
    center_lon = float(df_clean[lon_col].mean())

    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles="OpenStreetMap",
    )

    # Add satellite layer
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
        overlay=False,
        control=True,
    ).add_to(m)

    # Prepare values for coloring
    if value_col and value_col in df_clean.columns:
        values = df_clean[value_col].values
        valid_values = values[~np.isnan(values)]
        if len(valid_values) > 0:
            vmin = float(valid_values.min())
            vmax = float(valid_values.max())
        else:
            vmin = 0.0
            vmax = 1.0
    else:
        values = np.ones(len(df_clean))
        vmin = 0.0
        vmax = 1.0

    # Add well markers
    well_data = []
    for idx, row in df_clean.iterrows():
        lat = float(row[lat_col])
        lon = float(row[lon_col])

        if value_col and value_col in row.index:
            val = float(row[value_col]) if pd.notna(row[value_col]) else 0.0
        else:
            val = 1.0

        well_id = (
            str(row[well_id_col])
            if well_id_col and well_id_col in row.index
            else f"Well {idx}"
        )

        # Color code by value
        val_norm = (val - vmin) / (vmax - vmin + 1e-10)
        if val_norm < 0.33:
            color = "green"
        elif val_norm < 0.67:
            color = "orange"
        else:
            color = "red"

        well_data.append([lat, lon, val])

        popup_text = f"Well: {well_id}"
        if value_col and value_col in row.index:
            popup_text += f"<br>Production: {val:.2f}"

        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            popup=popup_text,
            tooltip=f"{well_id}: {val:.2f}" if value_col else well_id,
            color="black",
            fillColor=color,
            fillOpacity=0.7,
            weight=1,
        ).add_to(m)

    # Add heatmap if values provided
    if value_col and len(well_data) > 0 and HeatMap is not None:
        HeatMap(well_data, radius=15, blur=10, max_zoom=1).add_to(m)

    # Add legend
    legend_html = f"""
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 200px; height: 150px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <h4>{title}</h4>
    <p><b>Wells Shown:</b> {len(df_clean):,}</p>
    <p style="font-size:10px">Green = Low<br>
    Orange = Medium<br>
    Red = High</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Add layer control
    folium.LayerControl().add_to(m)

    # Save if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(output_path))
        logger.info(f"Interactive well map saved to: {output_path}")

    return m


def create_combined_map(
    datasets: List[Dict[str, Any]],
    output_file: Optional[Union[str, Path]] = None,
    zoom_start: int = 7,
) -> Optional[Any]:
    """Create a combined map with multiple layers.

    Uses Folium for web-based interactive maps.

    Args:
        datasets: List of dictionaries with keys:
            - 'coordinates': Well coordinates (n_wells, 2) - [lon, lat]
            - 'values': Production values (n_wells,)
            - 'name': Layer name
            - 'color': Optional marker color (default: blue, red, green, purple)
        output_file: Optional output file path.
        zoom_start: Initial zoom level, default 7.

    Returns:
        Folium map object, or None if folium not available.

    Example:
        >>> from geosmith.workflows.plotting import create_combined_map
        >>> import numpy as np
        >>>
        >>> datasets = [
        ...     {
        ...         'coordinates': np.array([[-120, 38], [-119, 39]]),
        ...         'values': np.array([1000, 2000]),
        ...         'name': 'Oil Production'
        ...     }
        ... ]
        >>> m = create_combined_map(datasets)
        >>> if m:
        ...     m.save('combined_map.html')
    """
    if not FOLIUM_AVAILABLE:
        logger.warning("folium not available, cannot create interactive map")
        return None

    if len(datasets) == 0:
        logger.warning("No datasets provided")
        return None

    # Calculate center from first dataset
    coords = datasets[0].get("coordinates", np.array([[0, 0]]))
    if coords.ndim == 2 and len(coords) > 0:
        center_lat = float(np.mean([coords[:, 1].min(), coords[:, 1].max()]))
        center_lon = float(np.mean([coords[:, 0].min(), coords[:, 0].max()]))
    else:
        center_lat = 0.0
        center_lon = 0.0

    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles="OpenStreetMap",
    )

    # Add satellite layer
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
        overlay=False,
        control=True,
    ).add_to(m)

    # Default colors
    default_colors = ["blue", "red", "green", "purple", "orange", "darkred", "lightred"]

    # Add each dataset as a layer
    for i, dataset in enumerate(datasets):
        coordinates = np.asarray(dataset.get("coordinates", np.array([[0, 0]])))
        values = np.asarray(dataset.get("values", np.ones(len(coordinates))))
        name = dataset.get("name", f"Layer {i+1}")
        color = dataset.get("color", default_colors[i % len(default_colors)])
        max_samples = dataset.get("max_samples", 500)

        if len(coordinates) == 0:
            continue

        n_samples = min(max_samples, len(coordinates))
        if n_samples < len(coordinates):
            indices = np.random.choice(len(coordinates), n_samples, replace=False)
        else:
            indices = np.arange(len(coordinates))

        feature_group = folium.FeatureGroup(name=name)

        for idx in indices:
            if coordinates[idx].shape[0] >= 2:
                lon = float(coordinates[idx, 0])
                lat = float(coordinates[idx, 1])
                val = (
                    float(values[idx])
                    if idx < len(values) and not np.isnan(values[idx])
                    else 0.0
                )

                if not np.isnan(lon) and not np.isnan(lat):
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=3,
                        popup=f"{name}: {val:.2f}",
                        tooltip=f"{name}: {val:.2f}",
                        color=color,
                        fillColor=color,
                        fillOpacity=0.6,
                        weight=1,
                    ).add_to(feature_group)

        feature_group.add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Save if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(output_path))
        logger.info(f"Combined map saved to: {output_path}")

    return m


# =============================================================================
# Geospatial Maps (Folium & GeoPandas)
# =============================================================================

# Optional GeoPandas dependency
try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon

    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    gpd = None  # type: ignore
    Point = None  # type: ignore
    Polygon = None  # type: ignore
    logger.debug(
        "geopandas not available. Static maps require geopandas. "
        "Install with: pip install geopandas"
    )


def create_field_map(
    wells_df: pd.DataFrame,
    x_col: str = "X",
    y_col: str = "Y",
    well_name_col: Optional[str] = None,
    color_by: Optional[str] = None,
    title: str = "Field Map",
    interactive: bool = True,
    basemap: str = "OpenStreetMap",
    use_plotsmith: Optional[bool] = None,
) -> Any:
    """Create an interactive or static field map showing well locations.

    Uses Folium for interactive maps or GeoPandas with PlotSmith styling
    for static maps.

    Args:
        wells_df: DataFrame with well coordinates and optional attributes.
        x_col: Column name for X coordinate (longitude or UTM X), default 'X'.
        y_col: Column name for Y coordinate (latitude or UTM Y), default 'Y'.
        well_name_col: Optional column name for well names.
        color_by: Optional column name to use for coloring wells.
        title: Map title, default 'Field Map'.
        interactive: If True, creates interactive Folium map. If False,
            creates static GeoPandas plot, default True.
        basemap: Basemap style for Folium ('OpenStreetMap', 'Stamen Terrain',
            'CartoDB positron'), default 'OpenStreetMap'.
        use_plotsmith: If True, use PlotSmith styling for static maps
            (if available). If None, auto-detect.

    Returns:
        Folium map object (interactive) or matplotlib Figure (static).

    Example:
        >>> from geosmith.workflows.plotting import create_field_map
        >>> import pandas as pd
        >>>
        >>> df = pd.DataFrame({
        ...     'X': [-120.0, -119.5, -119.0],
        ...     'Y': [38.5, 39.0, 39.5],
        ...     'WellName': ['Well_1', 'Well_2', 'Well_3'],
        ...     'Production': [1000, 2000, 1500]
        ... })
        >>> m = create_field_map(
        ...     df, x_col='X', y_col='Y', well_name_col='WellName',
        ...     color_by='Production', interactive=True
        ... )
        >>> if m:
        ...     m.save('field_map.html')
    """
    if interactive:
        if not FOLIUM_AVAILABLE:
            raise ImportError(
                "Folium is required for interactive maps. "
                "Install with: pip install folium"
            )

        return _create_folium_map(
            wells_df, x_col, y_col, well_name_col, color_by, title, basemap
        )
    else:
        if not GEOPANDAS_AVAILABLE:
            raise ImportError(
                "GeoPandas is required for static maps. "
                "Install with: pip install geopandas"
            )

        return _create_geopandas_map(
            wells_df, x_col, y_col, well_name_col, color_by, title, use_plotsmith
        )


def _create_folium_map(
    wells_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    well_name_col: Optional[str],
    color_by: Optional[str],
    title: str,
    basemap: str,
) -> Any:
    """Create interactive Folium map."""
    if not FOLIUM_AVAILABLE:
        raise ImportError("Folium is required for interactive maps")

    # Determine center of map
    center_lat = float(wells_df[y_col].mean())
    center_lon = float(wells_df[x_col].mean())

    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles=basemap,
    )

    # Add title
    title_html = f'<h3 align="center" style="font-size:20px"><b>{title}</b></h3>'
    m.get_root().html.add_child(folium.Element(title_html))

    # Color scheme
    if color_by and color_by in wells_df.columns:
        if pd.api.types.is_numeric_dtype(wells_df[color_by]):
            # Continuous color scale
            min_val = float(wells_df[color_by].min())
            max_val = float(wells_df[color_by].max())
            n_colors = 10
        else:
            # Categorical colors
            unique_vals = wells_df[color_by].unique()
            default_colors = [
                "#8dd3c7",
                "#ffffb3",
                "#bebada",
                "#fb8072",
                "#80b1d3",
                "#fdb462",
                "#b3de69",
                "#fccde5",
            ]
            color_map = dict(zip(unique_vals, default_colors[: len(unique_vals)]))
    else:
        colors = ["blue"] * len(wells_df)

    # Add well markers
    for idx, row in wells_df.iterrows():
        lat = float(row[y_col])
        lon = float(row[x_col])

        well_name = (
            row[well_name_col]
            if well_name_col and well_name_col in row.index
            else f"Well {idx}"
        )

        # Determine color
        if color_by and color_by in wells_df.columns:
            if pd.api.types.is_numeric_dtype(wells_df[color_by]):
                val = float(row[color_by])
                val_norm = (val - min_val) / (max_val - min_val + 1e-10)
                color_idx = int(val_norm * (n_colors - 1))
                # Simple color gradient
                colors = ["#440154", "#31688e", "#35b779", "#fde725"]  # Viridis-like
                color = colors[min(color_idx // 3, len(colors) - 1)]
            else:
                color = color_map.get(row[color_by], "blue")
        else:
            color = "blue"

        # Create popup text
        popup_text = f"<b>{well_name}</b><br>"
        popup_text += f"Coordinates: ({lon:.2f}, {lat:.2f})<br>"

        if color_by:
            popup_text += f"{color_by}: {row[color_by]}<br>"

        # Add marker
        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            popup=folium.Popup(popup_text, max_width=200),
            color="black",
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            weight=2,
        ).add_to(m)

    # Add fullscreen and measure tools
    if plugins is not None:
        plugins.Fullscreen().add_to(m)
        plugins.MeasureControl().add_to(m)

    return m


def _create_geopandas_map(
    wells_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    well_name_col: Optional[str],
    color_by: Optional[str],
    title: str,
    use_plotsmith: Optional[bool] = None,
) -> Figure:
    """Create static GeoPandas map with PlotSmith styling if available."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required for static maps. "
            "Install with: pip install matplotlib"
        )

    if not GEOPANDAS_AVAILABLE or gpd is None or Point is None:
        raise ImportError(
            "GeoPandas is required for static maps. "
            "Install with: pip install geopandas"
        )

    # Auto-detect PlotSmith if not specified
    if use_plotsmith is None:
        use_plotsmith = _is_plotsmith_available()

    if use_plotsmith:
        _apply_plotsmith_style()

    # Create GeoDataFrame
    geometry = [Point(xy) for xy in zip(wells_df[x_col], wells_df[y_col])]
    gdf = gpd.GeoDataFrame(wells_df, geometry=geometry, crs="EPSG:4326")

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot wells
    if color_by and color_by in wells_df.columns:
        gdf.plot(
            ax=ax,
            column=color_by,
            marker="o",
            markersize=100,
            legend=True,
            legend_kwds={"label": color_by, "shrink": 0.8},
            cmap=(
                "viridis"
                if pd.api.types.is_numeric_dtype(wells_df[color_by])
                else "Set3"
            ),
            edgecolor="black",
            linewidth=1.5,
        )
    else:
        gdf.plot(
            ax=ax,
            marker="o",
            markersize=100,
            color="blue",
            edgecolor="black",
            linewidth=1.5,
        )

    # Add well labels
    if well_name_col:
        for idx, row in gdf.iterrows():
            ax.annotate(
                text=row[well_name_col],
                xy=(row.geometry.x, row.geometry.y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude / X", fontsize=12)
    ax.set_ylabel("Latitude / Y", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Clean styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig


def create_well_trajectory_map(
    trajectories: List[Dict[str, Any]],
    title: str = "Well Trajectories",
    interactive: bool = True,
    use_plotsmith: Optional[bool] = None,
) -> Any:
    """Create a map showing well trajectories in 3D or 2D projection.

    Uses Plotly for interactive 3D plots or matplotlib with PlotSmith styling
    for static 2D plots.

    Args:
        trajectories: List of trajectory dictionaries, each containing:
            - 'name': Well name
            - 'x', 'y', 'z': Arrays of coordinates
            - Optional: 'color', 'log_values', 'facies'
        title: Map title, default 'Well Trajectories'.
        interactive: If True, creates interactive 3D Plotly plot. If False,
            creates 2D matplotlib plot, default True.
        use_plotsmith: If True, use PlotSmith styling for static plots
            (if available). If None, auto-detect.

    Returns:
        Plotly Figure (interactive 3D) or matplotlib Figure (static 2D).

    Example:
        >>> from geosmith.workflows.plotting import create_well_trajectory_map
        >>> import numpy as np
        >>>
        >>> trajectories = [
        ...     {
        ...         'name': 'Well_1',
        ...         'x': np.linspace(0, 1000, 100),
        ...         'y': np.linspace(0, 500, 100),
        ...         'z': np.linspace(0, 2000, 100),
        ...         'color': 'blue'
        ...     }
        ... ]
        >>> fig = create_well_trajectory_map(trajectories, interactive=True)
        >>> if hasattr(fig, 'show'):
        ...     fig.show()
    """
    if interactive:
        # Optional Plotly dependency
        try:
            import plotly.graph_objects as go

            PLOTLY_AVAILABLE = True
        except ImportError:
            PLOTLY_AVAILABLE = False
            go = None  # type: ignore

        if not PLOTLY_AVAILABLE:
            raise ImportError(
                "Plotly is required for interactive 3D trajectories. "
                "Install with: pip install plotly"
            )

        return _create_3d_trajectory_map(trajectories, title)
    else:
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError(
                "matplotlib is required for static trajectory maps. "
                "Install with: pip install matplotlib"
            )

        return _create_2d_trajectory_map(trajectories, title, use_plotsmith)


def _create_3d_trajectory_map(trajectories: List[Dict[str, Any]], title: str) -> Any:
    """Create interactive 3D trajectory map with Plotly."""
    try:
        import plotly.graph_objects as go

        PLOTLY_AVAILABLE = True
    except ImportError:
        raise ImportError("Plotly is required for 3D trajectories")

    fig = go.Figure()

    for traj in trajectories:
        name = traj.get("name", "Well")
        x = np.asarray(traj.get("x", []))
        y = np.asarray(traj.get("y", []))
        z = np.asarray(traj.get("z", []))
        color = traj.get("color", "blue")

        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines+markers",
                name=name,
                line=dict(color=color, width=4),
                marker=dict(size=3, color=color),
                hovertemplate=f"<b>{name}</b><br>"
                + "X: %{x:.2f}<br>"
                + "Y: %{y:.2f}<br>"
                + "Z: %{z:.2f}<br>"
                + "<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Depth (m)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        ),
        width=1200,
        height=900,
    )

    return fig


def _create_2d_trajectory_map(
    trajectories: List[Dict[str, Any]],
    title: str,
    use_plotsmith: Optional[bool] = None,
) -> Figure:
    """Create static 2D trajectory map with matplotlib and PlotSmith styling."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for static maps")

    # Auto-detect PlotSmith if not specified
    if use_plotsmith is None:
        use_plotsmith = _is_plotsmith_available()

    if use_plotsmith:
        _apply_plotsmith_style()

    fig, ax = plt.subplots(figsize=(12, 10))

    for traj in trajectories:
        name = traj.get("name", "Well")
        x = np.asarray(traj.get("x", []))
        y = np.asarray(traj.get("y", []))
        color = traj.get("color", "blue")

        ax.plot(x, y, marker="o", markersize=4, linewidth=2, label=name, color=color)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Clean styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig


# =============================================================================
# 3D Visualization (Plotly)
# =============================================================================

# Optional Plotly dependency
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None  # type: ignore
    px = None  # type: ignore
    make_subplots = None  # type: ignore
    logger.debug(
        "Plotly not available. Interactive 3D visualization requires Plotly. "
        "Install with: pip install plotly"
    )


def create_3d_well_log_viewer(
    df: pd.DataFrame,
    depth_col: str = "DEPTH",
    log_cols: Optional[List[str]] = None,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    z_col: Optional[str] = None,
    facies_col: Optional[str] = None,
    well_name: Optional[str] = None,
    title: Optional[str] = None,
) -> Any:
    """Create an interactive 3D well log viewer.

    Uses Plotly for 3D visualization. PlotSmith styling is not applicable
    for Plotly plots as they use HTML/JavaScript.

    Displays well logs in 3D space with depth as Z-axis and log values
    as X/Y axes or as color-coded traces along the wellbore.

    Args:
        df: Well log DataFrame.
        depth_col: Column name for depth, default 'DEPTH'.
        log_cols: List of log columns to visualize. If None, uses common logs.
        x_col: Optional column name for X coordinate (if well has spatial coordinates).
        y_col: Optional column name for Y coordinate (if well has spatial coordinates).
        z_col: Optional column name for Z coordinate (alternative to depth_col).
        facies_col: Optional column name for facies (for color coding).
        well_name: Optional name of the well.
        title: Optional plot title.

    Returns:
        Plotly Figure object (interactive 3D).

    Example:
        >>> from geosmith.workflows.plotting import create_3d_well_log_viewer
        >>> import pandas as pd
        >>>
        >>> df = pd.DataFrame({
        ...     'DEPTH': np.linspace(0, 1000, 100),
        ...     'GR': np.random.normal(60, 15, 100),
        ...     'RHOB': np.random.normal(2.5, 0.2, 100),
        ...     'NPHI': np.random.normal(0.2, 0.05, 100)
        ... })
        >>> fig = create_3d_well_log_viewer(df, log_cols=['GR', 'RHOB', 'NPHI'])
        >>> if hasattr(fig, 'show'):
        ...     fig.show()
    """
    if not PLOTLY_AVAILABLE or go is None:
        raise ImportError(
            "Plotly is required for 3D visualization. "
            "Install with: pip install plotly"
        )

    if log_cols is None:
        common_logs = ["GR", "RHOB", "NPHI", "RT", "PE"]
        log_cols = [col for col in common_logs if col in df.columns][:3]

    if not log_cols:
        raise ValueError("No log columns found to visualize")

    # Use depth or z coordinate
    if z_col and z_col in df.columns:
        z_values = df[z_col].values
    else:
        z_values = df[depth_col].values

    # Get spatial coordinates if available
    has_spatial = x_col and y_col and x_col in df.columns and y_col in df.columns

    fig = go.Figure()

    # Create 3D visualization
    if has_spatial:
        # 3D wellbore trajectory with log values
        x_coords = df[x_col].values
        y_coords = df[y_col].values

        # Color by first log or facies
        if facies_col and facies_col in df.columns:
            color_values = df[facies_col].values
            colorbar_title = "Facies"
        else:
            color_values = df[log_cols[0]].values
            colorbar_title = log_cols[0]

        fig.add_trace(
            go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_values,
                mode="markers+lines",
                marker=dict(
                    size=3,
                    color=color_values,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title=colorbar_title),
                ),
                line=dict(color="gray", width=2),
                name=well_name or "Well",
                hovertemplate="<b>%{text}</b><br>"
                + "X: %{x:.2f}<br>"
                + "Y: %{y:.2f}<br>"
                + "Z: %{z:.2f}<br>"
                + "<extra></extra>",
                text=[f"{log_cols[0]}: {v:.2f}" for v in df[log_cols[0]].values],
            )
        )
    else:
        # 2D projection: depth vs log values
        for i, log_col in enumerate(log_cols):
            if log_col not in df.columns:
                continue

            log_values = df[log_col].values

            # Create 3D trace with offset for multiple logs
            x_offset = i * 50  # Offset each log track

            fig.add_trace(
                go.Scatter3d(
                    x=[x_offset] * len(z_values),
                    y=log_values,
                    z=z_values,
                    mode="lines+markers",
                    name=log_col,
                    line=dict(width=3),
                    marker=dict(size=2),
                    hovertemplate=f"<b>{log_col}</b><br>"
                    + f"Value: %{{y:.2f}}<br>"
                    + f"Depth: %{{z:.2f}}<br>"
                    + "<extra></extra>",
                )
            )

    # Update layout
    title_text = title or (
        f"3D Well Log Viewer - {well_name}" if well_name else "3D Well Log Viewer"
    )

    fig.update_layout(
        title=title_text,
        scene=dict(
            xaxis_title="X" if has_spatial else "Log Track",
            yaxis_title="Y" if has_spatial else "Log Value",
            zaxis_title="Depth (m)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        ),
        width=1000,
        height=800,
    )

    return fig


def create_multi_well_3d_viewer(
    wells_data: List[Dict[str, Any]],
    depth_col: str = "DEPTH",
    x_col: str = "X",
    y_col: str = "Y",
    log_col: Optional[str] = None,
    facies_col: Optional[str] = None,
    title: str = "Multi-Well 3D Viewer",
) -> Any:
    """Create an interactive 3D viewer for multiple wells.

    Uses Plotly for 3D visualization.

    Displays multiple wells in 3D space with their trajectories
    and log values or facies color-coded.

    Args:
        wells_data: List of dictionaries, each containing:
            - 'df': DataFrame with well log data
            - 'name': Well name (optional)
            - 'x', 'y': Well coordinates (optional, if not in DataFrame)
        depth_col: Column name for depth, default 'DEPTH'.
        x_col: Column name for X coordinate, default 'X'.
        y_col: Column name for Y coordinate, default 'Y'.
        log_col: Optional log column to use for color coding.
        facies_col: Optional facies column to use for color coding.
        title: Plot title, default 'Multi-Well 3D Viewer'.

    Returns:
        Plotly Figure object (interactive 3D).

    Example:
        >>> from geosmith.workflows.plotting import create_multi_well_3d_viewer
        >>> import pandas as pd
        >>>
        >>> wells_data = [
        ...     {
        ...         'df': df1,
        ...         'name': 'Well_1',
        ...         'x': 1000,
        ...         'y': 2000
        ...     },
        ...     {
        ...         'df': df2,
        ...         'name': 'Well_2',
        ...         'x': 1500,
        ...         'y': 2500
        ...     }
        ... ]
        >>> fig = create_multi_well_3d_viewer(wells_data, log_col='GR')
        >>> if hasattr(fig, 'show'):
        ...     fig.show()
    """
    if not PLOTLY_AVAILABLE or go is None:
        raise ImportError(
            "Plotly is required for 3D visualization. "
            "Install with: pip install plotly"
        )

    fig = go.Figure()

    # Color scale for facies
    facies_colors = {
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

    for well_idx, well_info in enumerate(wells_data):
        df = well_info["df"]
        well_name = well_info.get("name", f"Well {well_idx + 1}")

        # Get coordinates
        if x_col in df.columns and y_col in df.columns:
            x_coords = df[x_col].values
            y_coords = df[y_col].values
        else:
            x_coords = np.full(len(df), well_info.get("x", well_idx * 100))
            y_coords = np.full(len(df), well_info.get("y", 0))

        z_values = df[depth_col].values

        # Determine color coding
        if facies_col and facies_col in df.columns:
            color_values = df[facies_col].values
            colors = [facies_colors.get(str(f), "#808080") for f in color_values]
            show_colorbar = False
        elif log_col and log_col in df.columns:
            color_values = df[log_col].values
            colors = color_values
            show_colorbar = True
        else:
            colors = (
                f"rgb({well_idx * 50 % 255}, "
                f"{well_idx * 100 % 255}, {well_idx * 150 % 255})"
            )
            show_colorbar = False

        fig.add_trace(
            go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_values,
                mode="markers+lines",
                marker=dict(
                    size=4,
                    color=colors,
                    showscale=show_colorbar,
                    colorbar=dict(title=log_col) if show_colorbar else None,
                    colorscale="Viridis" if show_colorbar else None,
                ),
                line=dict(color=colors if isinstance(colors, str) else "gray", width=3),
                name=well_name,
                hovertemplate=f"<b>{well_name}</b><br>"
                + "X: %{x:.2f}<br>"
                + "Y: %{y:.2f}<br>"
                + "Depth: %{z:.2f}<br>"
                + "<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Depth (m)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        ),
        width=1200,
        height=900,
    )

    return fig


def create_cross_section_viewer(
    wells_data: List[Dict[str, Any]],
    section_azimuth: float = 0.0,
    depth_col: str = "DEPTH",
    log_col: str = "GR",
    facies_col: Optional[str] = None,
    title: str = "Cross Section Viewer",
) -> Any:
    """Create an interactive cross section viewer.

    Uses Plotly for interactive visualization.

    Displays a 2D cross section through multiple wells along a specified
    azimuth, showing log values or facies.

    Args:
        wells_data: List of dictionaries, each containing:
            - 'df': DataFrame with well log data
            - 'name': Well name (optional)
            - 'x', 'y': Well coordinates
        section_azimuth: Azimuth of the cross section in degrees
            (0 = North), default 0.0.
        depth_col: Column name for depth, default 'DEPTH'.
        log_col: Log column to display, default 'GR'.
        facies_col: Optional facies column for color coding.
        title: Plot title, default 'Cross Section Viewer'.

    Returns:
        Plotly Figure object (interactive).

    Example:
        >>> from geosmith.workflows.plotting import create_cross_section_viewer
        >>> import pandas as pd
        >>>
        >>> wells_data = [
        ...     {
        ...         'df': df1,
        ...         'name': 'Well_1',
        ...         'x': 1000,
        ...         'y': 2000
        ...     },
        ...     {
        ...         'df': df2,
        ...         'name': 'Well_2',
        ...         'x': 2000,
        ...         'y': 2500
        ...     }
        ... ]
        >>> fig = create_cross_section_viewer(
        ...     wells_data, section_azimuth=45.0, log_col='GR'
        ... )
        >>> if hasattr(fig, 'show'):
        ...     fig.show()
    """
    if not PLOTLY_AVAILABLE or go is None:
        raise ImportError(
            "Plotly is required for cross section visualization. "
            "Install with: pip install plotly"
        )

    fig = go.Figure()

    # Calculate distances along cross section
    azimuth_rad = np.radians(section_azimuth)

    for well_info in wells_data:
        df = well_info["df"]
        well_name = well_info.get("name", "Well")

        x_well = well_info.get("x", 0)
        y_well = well_info.get("y", 0)

        # Project well onto cross section line
        distance = x_well * np.cos(azimuth_rad) + y_well * np.sin(azimuth_rad)

        z_values = df[depth_col].values

        if log_col in df.columns:
            log_values = df[log_col].values

            # Color by facies if available
            if facies_col and facies_col in df.columns:
                colors = df[facies_col].values
                fig.add_trace(
                    go.Scatter(
                        x=[distance] * len(z_values),
                        y=z_values,
                        mode="markers",
                        marker=dict(
                            size=5,
                            color=colors,
                            colorscale="Viridis",
                            showscale=True,
                            colorbar=dict(title="Facies"),
                        ),
                        name=well_name,
                        hovertemplate=f"<b>{well_name}</b><br>"
                        + f"{log_col}: %{{text}}<br>"
                        + "Depth: %{y:.2f}<br>"
                        + "<extra></extra>",
                        text=log_values,
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=[distance] * len(z_values),
                        y=z_values,
                        mode="lines+markers",
                        marker=dict(
                            size=3,
                            color=log_values,
                            colorscale="Viridis",
                            showscale=True,
                            colorbar=dict(title=log_col),
                        ),
                        name=well_name,
                        hovertemplate=f"<b>{well_name}</b><br>"
                        + f"{log_col}: %{{text}}<br>"
                        + "Depth: %{y:.2f}<br>"
                        + "<extra></extra>",
                        text=log_values,
                    )
                )

    fig.update_layout(
        title=title,
        xaxis_title="Distance along cross section (m)",
        yaxis_title="Depth (m)",
        yaxis=dict(autorange="reversed"),  # Invert depth axis
        width=1200,
        height=800,
    )

    return fig
