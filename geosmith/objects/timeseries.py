"""Time series compatible objects using TimeSmith's typing layer.

This module re-exports TimeSmith's SeriesLike, PanelLike, and TableLike
from timesmith.typing, ensuring compatibility across the *Smith ecosystem.

TimeSmith is a required dependency for time series functionality.
"""

# Import from timesmith.typing (single source of truth)
try:
    from timesmith.typing import PanelLike, SeriesLike, TableLike
    from timesmith.typing.validators import (
        assert_panel_like,
        assert_series_like,
        assert_table,
    )

    # Alias assert_table as assert_table_like for consistency
    assert_table_like = assert_table
    
    __all__ = [
        "PanelLike",
        "SeriesLike",
        "TableLike",
        "assert_panel_like",
        "assert_series_like",
        "assert_table_like",
    ]
except ImportError as e:
    raise ImportError(
        "timesmith is required for time series objects. "
        "Install with: pip install geosmith[timesmith] or pip install timesmith"
    ) from e
