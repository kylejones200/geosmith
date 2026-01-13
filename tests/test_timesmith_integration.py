"""Integration tests for TimeSmith typing compatibility.

Tests that GeoSmith correctly uses timesmith.typing as the single source of truth.
"""

import numpy as np
import pandas as pd

# Import from timesmith.typing (single source of truth)
from timesmith.typing.validators import (
    assert_panel_like,
    assert_series_like,
    assert_table,
)

# Note: SeriesLike, PanelLike, TableLike are Protocols, not classes
# They're imported from timesmith.typing and used for type checking


class TestTimeSmithIntegration:
    """Tests for TimeSmith typing integration."""

    def test_serieslike_is_same_type(self):
        """Test that SeriesLike Protocol works with pandas Series."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        series = pd.Series(np.random.randn(10), index=dates)

        # SeriesLike is a Protocol - pandas Series should conform to it
        assert_series_like(series)
        # Both should be the same pandas Series type
        assert isinstance(series, pd.Series)

    def test_serieslike_validator_works(self):
        """Test that TimeSmith validators work with pandas Series."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        series = pd.Series(np.random.randn(10), index=dates)

        # TimeSmith validator should work directly on pandas Series
        assert_series_like(series)

    def test_panellike_is_same_type(self):
        """Test that PanelLike Protocol works with pandas DataFrame."""
        panel_data = pd.DataFrame(
            {
                "entity": ["A", "A", "B", "B"],
                "value": np.random.randn(4),
            },
            index=pd.date_range("2020-01-01", periods=4, freq="D"),
        )

        # PanelLike is a Protocol - pandas DataFrame should conform to it
        assert_panel_like(panel_data)
        assert isinstance(panel_data, pd.DataFrame)

    def test_panellike_validator_works(self):
        """Test that TimeSmith validators work with pandas DataFrame."""
        panel_data = pd.DataFrame(
            {
                "entity": ["A", "A", "B", "B"],
                "value": np.random.randn(4),
            },
            index=pd.date_range("2020-01-01", periods=4, freq="D"),
        )

        # TimeSmith validator should work directly on pandas DataFrame
        assert_panel_like(panel_data)

    def test_tablelike_is_same_type(self):
        """Test that TableLike Protocol works with pandas DataFrame."""
        table_data = pd.DataFrame(
            {
                "feature1": np.random.randn(10),
                "feature2": np.random.randn(10),
            },
            index=pd.date_range("2020-01-01", periods=10, freq="D"),
        )

        # TableLike is a Protocol - pandas DataFrame should conform to it
        assert_table(table_data)
        assert isinstance(table_data, pd.DataFrame)

    def test_tablelike_validator_works(self):
        """Test that TimeSmith validators work with pandas DataFrame."""
        table_data = pd.DataFrame(
            {
                "feature1": np.random.randn(10),
                "feature2": np.random.randn(10),
            },
            index=pd.date_range("2020-01-01", periods=10, freq="D"),
        )

        # TimeSmith validator should work directly on pandas DataFrame
        assert_table(table_data)

    def test_no_circular_imports(self):
        """Test that there are no circular imports."""
        # This test passes if imports succeed
        from geosmith import PanelLike, PointSet, SeriesLike, TableLike

        assert PointSet is not None
        assert SeriesLike is not None
        assert PanelLike is not None
        assert TableLike is not None

