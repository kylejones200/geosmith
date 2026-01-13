"""Compatibility helpers for TimeSmith integration.

This module provides helper functions for working with TimeSmith's typing layer.
TimeSmith is a required dependency for time series functionality.
"""

from typing import Union

import pandas as pd

from timesmith.typing import PanelLike, SeriesLike, TableLike
from timesmith.typing.validators import (
    assert_panel_like,
    assert_series_like,
    assert_table_like,
)


def validate_series_like(series: Union[SeriesLike, pd.Series]) -> SeriesLike:
    """Validate and return SeriesLike object.

    Args:
        series: SeriesLike or pandas Series.

    Returns:
        SeriesLike object (validated pandas Series or DataFrame).

    Raises:
        TypeError: If validation fails.
    """
    # SeriesLike is a Protocol, so we validate and return the pandas object
    assert_series_like(series)
    return series  # type: ignore[return-value]


def validate_panel_like(
    panel: Union[PanelLike, pd.DataFrame], entity_col: str = "entity"
) -> PanelLike:
    """Validate and return PanelLike object.

    Args:
        panel: PanelLike or pandas DataFrame.
        entity_col: Name of entity column (unused, kept for compatibility).

    Returns:
        PanelLike object (validated pandas DataFrame).

    Raises:
        TypeError: If validation fails.
    """
    # PanelLike is a Protocol, so we validate and return the pandas DataFrame
    assert_panel_like(panel)
    return panel  # type: ignore[return-value]


def validate_table_like(
    table: Union[TableLike, pd.DataFrame], time_col: str | None = None
) -> TableLike:
    """Validate and return TableLike object.

    Args:
        table: TableLike or pandas DataFrame.
        time_col: Name of time column (unused, kept for compatibility).

    Returns:
        TableLike object (validated pandas DataFrame).

    Raises:
        TypeError: If validation fails.
    """
    # TableLike is a Protocol, so we validate and return the pandas DataFrame
    assert_table_like(table)
    return table  # type: ignore[return-value]
