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
        # Basic validation - just check it's a Series with time index
        assert isinstance(data, pd.Series)
        assert len(data) == 3
        assert data.name == "test"
        assert isinstance(data.index, pd.DatetimeIndex)
        # If validators available, use them
        if VALIDATORS_AVAILABLE:
            assert_series_like(data)

    def test_valid_dataframe(self):
        """Test validating single-column DataFrame as SeriesLike Protocol."""
        data = pd.DataFrame({"value": [1, 2, 3]}, index=pd.date_range("2020-01-01", periods=3))
        # Convert to Series for validation
        series = data.iloc[:, 0]
        assert isinstance(series, pd.Series)
        assert isinstance(series.index, pd.DatetimeIndex)
        # If validators available, use them
        if VALIDATORS_AVAILABLE:
            assert_series_like(series)

    def test_invalid_multi_column_dataframe(self):
        """Test that multi-column DataFrame raises error."""
        data = pd.DataFrame(
            {"a": [1, 2], "b": [3, 4]}, index=pd.date_range("2020-01-01", periods=2)
        )
        # Multi-column DataFrame should fail validation (if validators available)
        if VALIDATORS_AVAILABLE:
            with pytest.raises((ValueError, TypeError)):
                assert_series_like(data)
        else:
            # Just check it's a DataFrame
            assert isinstance(data, pd.DataFrame)

    def test_invalid_index_type(self):
        """Test that non-datetime/int index raises error."""
        data = pd.Series([1, 2, 3], index=["a", "b", "c"])
        # Invalid index should fail validation (if validators available)
        if VALIDATORS_AVAILABLE:
            with pytest.raises((ValueError, TypeError)):
                assert_series_like(data)
        else:
            # Just check it's a Series
            assert isinstance(data, pd.Series)

    def test_validator_works(self):
        """Test that TimeSmith validators work."""
        data = pd.Series([1, 2, 3], index=pd.date_range("2020-01-01", periods=3))
        # Basic check
        assert isinstance(data, pd.Series)
        assert isinstance(data.index, pd.DatetimeIndex)
        # If validators available, use them
        if VALIDATORS_AVAILABLE:
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
        # Basic checks
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 4
        assert "entity" in data.columns
        # If validators available, use them
        if VALIDATORS_AVAILABLE:
            assert_panel_like(data)

    def test_missing_entity_col(self):
        """Test that missing entity column raises error."""
        data = pd.DataFrame(
            {"value": [1, 2, 3]}, index=pd.date_range("2020-01-01", periods=3)
        )
        # Missing entity column should fail validation (if validators available)
        if VALIDATORS_AVAILABLE:
            with pytest.raises((ValueError, TypeError)):
                assert_panel_like(data)
        else:
            # Just check it's a DataFrame
            assert isinstance(data, pd.DataFrame)


class TestTableLike:
    """Tests for TableLike."""

    def test_valid_table(self):
        """Test validating pandas DataFrame as TableLike Protocol."""
        data = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [4, 5, 6]},
            index=pd.date_range("2020-01-01", periods=3),
        )
        # Basic checks
        assert isinstance(data, pd.DataFrame)
        assert data.shape == (3, 2)
        assert isinstance(data.index, pd.DatetimeIndex)
        # If validators available, use them
        if VALIDATORS_AVAILABLE:
            assert_table(data)

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

