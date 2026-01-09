"""Tests for time series objects compatible with TimeSmith."""

import pandas as pd
import pytest

from geosmith.objects.timeseries import PanelLike, SeriesLike, TableLike


class TestSeriesLike:
    """Tests for SeriesLike."""

    def test_valid_series(self):
        """Test creating valid SeriesLike from Series."""
        data = pd.Series([1, 2, 3], index=pd.date_range("2020-01-01", periods=3))
        series = SeriesLike(data=data, name="test")
        assert len(series.data) == 3
        assert series.name == "test"

    def test_valid_dataframe(self):
        """Test creating SeriesLike from single-column DataFrame."""
        data = pd.DataFrame({"value": [1, 2, 3]}, index=pd.date_range("2020-01-01", periods=3))
        series = SeriesLike(data=data)
        assert isinstance(series.data, pd.Series)

    def test_invalid_multi_column_dataframe(self):
        """Test that multi-column DataFrame raises error."""
        data = pd.DataFrame(
            {"a": [1, 2], "b": [3, 4]}, index=pd.date_range("2020-01-01", periods=2)
        )
        with pytest.raises(ValueError, match="exactly 1 column"):
            SeriesLike(data=data)

    def test_invalid_index_type(self):
        """Test that non-datetime/int index raises error."""
        data = pd.Series([1, 2, 3], index=["a", "b", "c"])
        with pytest.raises(ValueError, match="Index must be DatetimeIndex"):
            SeriesLike(data=data)

    def test_to_timesmith_no_dependency(self):
        """Test to_timesmith when TimeSmith not available."""
        data = pd.Series([1, 2, 3], index=pd.date_range("2020-01-01", periods=3))
        series = SeriesLike(data=data)
        result = series.to_timesmith()
        # Should return self when TimeSmith not available
        assert result is series


class TestPanelLike:
    """Tests for PanelLike."""

    def test_valid_panel(self):
        """Test creating valid PanelLike."""
        data = pd.DataFrame(
            {
                "entity": ["A", "A", "B", "B"],
                "value": [1, 2, 3, 4],
            },
            index=pd.date_range("2020-01-01", periods=4),
        )
        panel = PanelLike(data=data, entity_col="entity")
        assert len(panel.data) == 4
        assert panel.entity_col == "entity"

    def test_missing_entity_col(self):
        """Test that missing entity column raises error."""
        data = pd.DataFrame(
            {"value": [1, 2, 3]}, index=pd.date_range("2020-01-01", periods=3)
        )
        with pytest.raises(ValueError, match="Entity column"):
            PanelLike(data=data, entity_col="entity")


class TestTableLike:
    """Tests for TableLike."""

    def test_valid_table(self):
        """Test creating valid TableLike."""
        data = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [4, 5, 6]},
            index=pd.date_range("2020-01-01", periods=3),
        )
        table = TableLike(data=data)
        assert table.data.shape == (3, 2)

    def test_table_with_time_col(self):
        """Test TableLike with explicit time column."""
        data = pd.DataFrame(
            {
                "time": pd.date_range("2020-01-01", periods=3),
                "feature1": [1, 2, 3],
            }
        )
        table = TableLike(data=data, time_col="time")
        assert table.time_col == "time"

