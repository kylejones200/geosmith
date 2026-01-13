"""Plotting helpers for Layer 1 objects.

Optionally uses PlotSmith when available for cleaner, publication-ready plots.
PlotSmith is NOT a hard dependency - falls back to matplotlib when not available.
"""

import logging
from typing import Optional

import numpy as np

from geosmith.objects.lineset import LineSet
from geosmith.objects.pointset import PointSet
from geosmith.objects.polygonset import PolygonSet
from geosmith.objects.rastergrid import RasterGrid

logger = logging.getLogger(__name__)


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


def plot_points(
    points: PointSet,
    ax=None,
    use_plotsmith: Optional[bool] = None,
    **kwargs,
):
    """Plot PointSet.

    Uses PlotSmith if available for cleaner plots, otherwise uses matplotlib.

    Args:
        points: PointSet to plot.
        ax: Optional matplotlib axes.
        use_plotsmith: If True, use PlotSmith (if available). If None, auto-detect.
        **kwargs: Additional arguments passed to scatter.

    Returns:
        matplotlib axes.
    """
    # Auto-detect PlotSmith if not specified
    if use_plotsmith is None:
        use_plotsmith = _is_plotsmith_available()

    if use_plotsmith and _is_plotsmith_available():
        try:
            from plotsmith import plot_scatter

            coords = points.coordinates[:, :2]
            if ax is None:
                fig, ax = plot_scatter(coords[:, 0], coords[:, 1], **kwargs)
            else:
                ax.scatter(coords[:, 0], coords[:, 1], **kwargs)
            return ax
        except ImportError:
            # Fall through to matplotlib
            pass

    # Fallback to matplotlib
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "plotting requires matplotlib. Install with: pip install geosmith[viz]"
        )

    if ax is None:
        ax = plt.gca()

    coords = points.coordinates[:, :2]
    ax.scatter(coords[:, 0], coords[:, 1], **kwargs)
    return ax


def plot_polygons(
    polygons: PolygonSet,
    ax=None,
    **kwargs,
):
    """Plot PolygonSet.

    Args:
        polygons: PolygonSet to plot.
        ax: Optional matplotlib axes.
        **kwargs: Additional arguments passed to plot.

    Returns:
        matplotlib axes.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MPLPolygon
    except ImportError:
        raise ImportError(
            "plotting requires matplotlib. Install with: pip install geosmith[viz]"
        )

    if ax is None:
        ax = plt.gca()

    for polygon_rings in polygons.rings:
        if not polygon_rings:
            continue
        exterior = polygon_rings[0]
        holes = polygon_rings[1:] if len(polygon_rings) > 1 else None

        # Plot exterior
        poly = MPLPolygon(exterior[:, :2], **kwargs)
        ax.add_patch(poly)

        # Plot holes
        if holes:
            for hole in holes:
                hole_poly = MPLPolygon(
                    hole[:, :2], facecolor="white", edgecolor="black"
                )
                ax.add_patch(hole_poly)

    return ax


def plot_raster(
    raster: RasterGrid,
    ax=None,
    band: int = 0,
    **kwargs,
):
    """Plot RasterGrid.

    Args:
        raster: RasterGrid to plot.
        ax: Optional matplotlib axes.
        band: Band index to plot.
        **kwargs: Additional arguments passed to imshow.

    Returns:
        matplotlib axes.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "plotting requires matplotlib. Install with: pip install geosmith[viz]"
        )

    if ax is None:
        ax = plt.gca()

    data = raster.data
    if data.ndim == 3:
        data = data[band, :, :]

    ax.imshow(data, **kwargs)
    return ax
