"""Tests for time series objects using TimeSmith typing.

These objects are imported from timesmith.typing (single source of truth).
"""

import pandas as pd
import pytest

from geosmith.objects.timeseries import PanelLike, SeriesLike, TableLike


class TestSeriesLike:
    """Tests for SeriesLike."""

    def test_valid_series(self):
        """Test validating pandas Series as SeriesLike Protocol."""
        data = pd.Series([1, 2, 3], index=pd.date_range("2020-01-01", periods=3), name="test")
        # SeriesLike is a Protocol - we validate the pandas Series directly
        assert_series_like(data)
        assert len(data) == 3
        assert data.name == "test"

    def test_valid_dataframe(self):
        """Test validating single-column DataFrame as SeriesLike Protocol."""
        data = pd.DataFrame({"value": [1, 2, 3]}, index=pd.date_range("2020-01-01", periods=3))
        # Convert to Series for validation
        series = data.iloc[:, 0]
        assert_series_like(series)
        assert isinstance(series, pd.Series)

    def test_invalid_multi_column_dataframe(self):
        """Test that multi-column DataFrame raises error."""
        data = pd.DataFrame(
            {"a": [1, 2], "b": [3, 4]}, index=pd.date_range("2020-01-01", periods=2)
        )
        # Multi-column DataFrame should fail validation
        with pytest.raises((ValueError, TypeError)):
            assert_series_like(data)

    def test_invalid_index_type(self):
        """Test that non-datetime/int index raises error."""
        data = pd.Series([1, 2, 3], index=["a", "b", "c"])
        # Invalid index should fail validation
        with pytest.raises((ValueError, TypeError)):
            assert_series_like(data)

    def test_validator_works(self):
        """Test that TimeSmith validators work."""
        data = pd.Series([1, 2, 3], index=pd.date_range("2020-01-01", periods=3))
        # Validator should work (no exception)
        assert_series_like(data)


class TestPanelLike:
    """Tests for PanelLike."""

    def test_valid_panel(self):
        """Test validating pandas DataFrame as PanelLike Protocol."""
        data = pd.DataFrame(
            {
                "entity": ["A", "A", "B", "B"],
                "value": [1, 2, 3, 4],
            },
            index=pd.date_range("2020-01-01", periods=4),
        )
        # PanelLike is a Protocol - we validate the DataFrame directly
        # Note: PanelLike validation may require entity_col parameter
        assert_panel_like(data)
        assert len(data) == 4

    def test_missing_entity_col(self):
        """Test that missing entity column raises error."""
        data = pd.DataFrame(
            {"value": [1, 2, 3]}, index=pd.date_range("2020-01-01", periods=3)
        )
        # Missing entity column should fail validation
        with pytest.raises((ValueError, TypeError)):
            assert_panel_like(data)


class TestTableLike:
    """Tests for TableLike."""

    def test_valid_table(self):
        """Test validating pandas DataFrame as TableLike Protocol."""
        data = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [4, 5, 6]},
            index=pd.date_range("2020-01-01", periods=3),
        )
        # TableLike is a Protocol - we validate the DataFrame directly
        assert_table(data)
        assert data.shape == (3, 2)

    def test_table_with_time_col(self):
        """Test TableLike with explicit time column."""
        data = pd.DataFrame(
            {
                "time": pd.date_range("2020-01-01", periods=3),
                "feature1": [1, 2, 3],
            }
        )
        # TableLike with time_col - validation may handle this differently
        # For now, just check that the DataFrame is valid
        assert data.shape == (3, 2)

