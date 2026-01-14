"""Petrophysics plotting utilities.

Layer 4: Workflows - Public entry points with plotting.
Migrated from geosuite.petro.buckles, geosuite.petro.pickett.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from geosmith.workflows.plotting._base import MATPLOTLIB_AVAILABLE, Figure
from geosmith.workflows.plotting._base import plt  # noqa: F401


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

