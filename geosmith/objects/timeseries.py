"""Time series compatible objects matching TimeSmith's typing layer.

These objects are compatible with TimeSmith's SeriesLike, PanelLike, and TableLike
types, allowing seamless integration without conversion when TimeSmith is available.
TimeSmith is NOT a hard dependency - these objects work independently.
"""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SeriesLike:
    """Time series data compatible with TimeSmith's SeriesLike type.

    Represents a pandas Series or single-column DataFrame with datetime/int index.
    Compatible with TimeSmith's typing layer for seamless integration.

    Attributes:
        data: pandas Series or single-column DataFrame with time index.
        name: Optional name for the series.
    """

    data: Union[pd.Series, pd.DataFrame]
    name: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate SeriesLike parameters."""
        if isinstance(self.data, pd.DataFrame):
            if len(self.data.columns) != 1:
                raise ValueError(
                    f"DataFrame must have exactly 1 column, got {len(self.data.columns)}"
                )
            # Convert to Series
            series = self.data.iloc[:, 0]
            object.__setattr__(self, "data", series)

        if not isinstance(self.data, pd.Series):
            raise ValueError(
                f"data must be pandas Series or single-column DataFrame, "
                f"got {type(self.data)}"
            )

        # Check index type (datetime or int)
        index = self.data.index
        if not (
            isinstance(index, pd.DatetimeIndex)
            or (isinstance(index, pd.Index) and index.dtype in ("int64", "int32"))
        ):
            raise ValueError(
                f"Index must be DatetimeIndex or integer Index, got {type(index)}"
            )

    def to_timesmith(self):
        """Convert to TimeSmith SeriesLike (if TimeSmith is available).

        Returns:
            TimeSmith SeriesLike object, or self if TimeSmith not available.
        """
        try:
            from timesmith.typing import SeriesLike as TSSeriesLike
            return TSSeriesLike(data=self.data, name=self.name)
        except ImportError:
            return self

    @classmethod
    def from_timesmith(cls, ts_series):
        """Create from TimeSmith SeriesLike object.

        Args:
            ts_series: TimeSmith SeriesLike object.

        Returns:
            GeoSmith SeriesLike object.
        """
        return cls(data=ts_series.data, name=ts_series.name)

    def __repr__(self) -> str:
        """String representation."""
        name_str = f", name='{self.name}'" if self.name else ""
        return f"SeriesLike(length={len(self.data)}{name_str})"


@dataclass(frozen=True)
class PanelLike:
    """Panel data compatible with TimeSmith's PanelLike type.

    Represents a DataFrame with entity key plus time index.
    Compatible with TimeSmith's typing layer for seamless integration.

    Attributes:
        data: DataFrame with entity key column and time index.
        entity_col: Name of the entity key column. Defaults to 'entity'.
        time_col: Name of the time column (if not in index). Defaults to None (uses index).
    """

    data: pd.DataFrame
    entity_col: str = "entity"
    time_col: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate PanelLike parameters."""
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError(
                f"data must be pandas DataFrame, got {type(self.data)}"
            )

        if self.entity_col not in self.data.columns:
            raise ValueError(
                f"Entity column '{self.entity_col}' not found in DataFrame. "
                f"Available columns: {list(self.data.columns)}"
            )

        # Check time index or time column
        if self.time_col is None:
            # Use index as time
            index = self.data.index
            if not (
                isinstance(index, pd.DatetimeIndex)
                or (isinstance(index, pd.Index) and index.dtype in ("int64", "int32"))
            ):
                raise ValueError(
                    f"When time_col is None, index must be DatetimeIndex or integer Index, "
                    f"got {type(index)}"
                )
        else:
            if self.time_col not in self.data.columns:
                raise ValueError(
                    f"Time column '{self.time_col}' not found in DataFrame. "
                    f"Available columns: {list(self.data.columns)}"
                )

    def to_timesmith(self):
        """Convert to TimeSmith PanelLike (if TimeSmith is available).

        Returns:
            TimeSmith PanelLike object, or self if TimeSmith not available.
        """
        try:
            from timesmith.typing import PanelLike as TSPanelLike
            return TSPanelLike(
                data=self.data, entity_col=self.entity_col, time_col=self.time_col
            )
        except ImportError:
            return self

    @classmethod
    def from_timesmith(cls, ts_panel):
        """Create from TimeSmith PanelLike object.

        Args:
            ts_panel: TimeSmith PanelLike object.

        Returns:
            GeoSmith PanelLike object.
        """
        return cls(
            data=ts_panel.data,
            entity_col=ts_panel.entity_col,
            time_col=ts_panel.time_col,
        )

    def __repr__(self) -> str:
        """String representation."""
        n_entities = self.data[self.entity_col].nunique()
        n_rows = len(self.data)
        return f"PanelLike(n_entities={n_entities}, n_rows={n_rows}, entity_col='{self.entity_col}')"


@dataclass(frozen=True)
class TableLike:
    """Table data compatible with TimeSmith's TableLike type.

    Represents a DataFrame with row index aligned to time.
    Compatible with TimeSmith's typing layer for seamless integration.

    Attributes:
        data: DataFrame with time-aligned row index.
        time_col: Name of the time column (if not in index). Defaults to None (uses index).
    """

    data: pd.DataFrame
    time_col: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate TableLike parameters."""
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError(
                f"data must be pandas DataFrame, got {type(self.data)}"
            )

        # Check time index or time column
        if self.time_col is None:
            # Use index as time
            index = self.data.index
            if not (
                isinstance(index, pd.DatetimeIndex)
                or (isinstance(index, pd.Index) and index.dtype in ("int64", "int32"))
            ):
                raise ValueError(
                    f"When time_col is None, index must be DatetimeIndex or integer Index, "
                    f"got {type(index)}"
                )
        else:
            if self.time_col not in self.data.columns:
                raise ValueError(
                    f"Time column '{self.time_col}' not found in DataFrame. "
                    f"Available columns: {list(self.data.columns)}"
                )

    def to_timesmith(self):
        """Convert to TimeSmith TableLike (if TimeSmith is available).

        Returns:
            TimeSmith TableLike object, or self if TimeSmith not available.
        """
        try:
            from timesmith.typing import TableLike as TSTableLike
            return TSTableLike(data=self.data, time_col=self.time_col)
        except ImportError:
            return self

    @classmethod
    def from_timesmith(cls, ts_table):
        """Create from TimeSmith TableLike object.

        Args:
            ts_table: TimeSmith TableLike object.

        Returns:
            GeoSmith TableLike object.
        """
        return cls(data=ts_table.data, time_col=ts_table.time_col)

    def __repr__(self) -> str:
        """String representation."""
        n_rows, n_cols = self.data.shape
        return f"TableLike(n_rows={n_rows}, n_cols={n_cols}, time_col={self.time_col})"

