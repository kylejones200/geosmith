"""Time series compatible objects using TimeSmith's typing layer.

This module re-exports TimeSmith's SeriesLike, PanelLike, and TableLike
from timesmith.typing, ensuring compatibility across the *Smith ecosystem.

TimeSmith is a required dependency for time series functionality.
"""

# Import from timesmith.typing (single source of truth)
# Note: timesmith may have optional dependencies (e.g., networkx) that cause
# AttributeError during import. We catch both ImportError and AttributeError.
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
except (ImportError, AttributeError) as e:
    # AttributeError can occur if timesmith has missing optional dependencies
    # (e.g., networkx) that are imported at module level
    raise ImportError(
        "timesmith is required for time series objects. "
        "Install with: pip install geosmith[timesmith] or pip install timesmith. "
        "If timesmith is installed but you see AttributeError, you may need to "
        "install optional dependencies like networkx."
    ) from e
