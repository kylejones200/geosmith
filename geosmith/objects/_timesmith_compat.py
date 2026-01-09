"""Compatibility layer for TimeSmith integration.

This module provides seamless conversion between GeoSmith and TimeSmith objects
when TimeSmith is available. TimeSmith is NOT a hard dependency.
"""

from typing import Optional, Union

import pandas as pd

from geosmith.objects.timeseries import PanelLike, SeriesLike, TableLike


def is_timesmith_available() -> bool:
    """Check if TimeSmith is available.

    Returns:
        True if TimeSmith can be imported, False otherwise.
    """
    try:
        import timesmith  # noqa: F401
        return True
    except ImportError:
        return False


def to_timesmith_serieslike(series: Union[SeriesLike, pd.Series]) -> "SeriesLike":
    """Convert to TimeSmith SeriesLike if available, otherwise return as-is.

    Args:
        series: GeoSmith SeriesLike or pandas Series.

    Returns:
        TimeSmith SeriesLike if available, otherwise GeoSmith SeriesLike.
    """
    if isinstance(series, SeriesLike):
        return series.to_timesmith()
    else:
        # Create GeoSmith SeriesLike first
        gs_series = SeriesLike(data=series)
        return gs_series.to_timesmith()


def to_timesmith_panellike(panel: Union[PanelLike, pd.DataFrame]) -> "PanelLike":
    """Convert to TimeSmith PanelLike if available, otherwise return as-is.

    Args:
        panel: GeoSmith PanelLike or pandas DataFrame.

    Returns:
        TimeSmith PanelLike if available, otherwise GeoSmith PanelLike.
    """
    if isinstance(panel, PanelLike):
        return panel.to_timesmith()
    else:
        # Create GeoSmith PanelLike first (assumes 'entity' column)
        gs_panel = PanelLike(data=panel, entity_col="entity")
        return gs_panel.to_timesmith()


def to_timesmith_tablelike(table: Union[TableLike, pd.DataFrame]) -> "TableLike":
    """Convert to TimeSmith TableLike if available, otherwise return as-is.

    Args:
        table: GeoSmith TableLike or pandas DataFrame.

    Returns:
        TimeSmith TableLike if available, otherwise GeoSmith TableLike.
    """
    if isinstance(table, TableLike):
        return table.to_timesmith()
    else:
        # Create GeoSmith TableLike first
        gs_table = TableLike(data=table)
        return gs_table.to_timesmith()


def from_timesmith(obj) -> Union[SeriesLike, PanelLike, TableLike]:
    """Convert from TimeSmith object to GeoSmith object.

    Args:
        obj: TimeSmith SeriesLike, PanelLike, or TableLike object.

    Returns:
        Corresponding GeoSmith object.
    """
    if not is_timesmith_available():
        raise ImportError(
            "TimeSmith is not available. Install with: pip install geosmith[timesmith]"
        )

    from timesmith.typing import PanelLike as TSPanelLike
    from timesmith.typing import SeriesLike as TSSeriesLike
    from timesmith.typing import TableLike as TSTableLike

    if isinstance(obj, TSSeriesLike):
        return SeriesLike.from_timesmith(obj)
    elif isinstance(obj, TSPanelLike):
        return PanelLike.from_timesmith(obj)
    elif isinstance(obj, TSTableLike):
        return TableLike.from_timesmith(obj)
    else:
        raise ValueError(
            f"Unknown TimeSmith object type: {type(obj)}. "
            f"Expected SeriesLike, PanelLike, or TableLike."
        )

