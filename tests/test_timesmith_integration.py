"""Integration tests for TimeSmith typing compatibility.

Tests that GeoSmith correctly uses timesmith.typing as the single source of truth.
"""

import numpy as np
import pandas as pd
import pytest

# Import from timesmith.typing (single source of truth)
from timesmith.typing import PanelLike, SeriesLike, TableLike
from timesmith.typing.validators import (
    assert_panel_like,
    assert_series_like,
    assert_table_like,
)

# Import GeoSmith objects (which re-export from timesmith.typing)
from geosmith.objects.timeseries import PanelLike as GSPanelLike
from geosmith.objects.timeseries import SeriesLike as GSSeriesLike
from geosmith.objects.timeseries import TableLike as GSTableLike


class TestTimeSmithIntegration:
    """Tests for TimeSmith typing integration."""

    def test_serieslike_is_same_type(self):
        """Test that GeoSmith SeriesLike is same as TimeSmith SeriesLike."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        series = pd.Series(np.random.randn(10), index=dates)

        ts_series = SeriesLike(data=series)
        gs_series = GSSeriesLike(data=series)

        # Should be same type (imported from timesmith.typing)
        assert type(ts_series) == type(gs_series)
        assert isinstance(gs_series, SeriesLike)

    def test_serieslike_validator_works(self):
        """Test that TimeSmith validators work with GeoSmith objects."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        series = pd.Series(np.random.randn(10), index=dates)

        gs_series = GSSeriesLike(data=series)

        # TimeSmith validator should work
        assert_series_like(gs_series)

    def test_panellike_is_same_type(self):
        """Test that GeoSmith PanelLike is same as TimeSmith PanelLike."""
        panel_data = pd.DataFrame(
            {
                "entity": ["A", "A", "B", "B"],
                "value": np.random.randn(4),
            },
            index=pd.date_range("2020-01-01", periods=4, freq="D"),
        )

        ts_panel = PanelLike(data=panel_data, entity_col="entity")
        gs_panel = GSPanelLike(data=panel_data, entity_col="entity")

        # Should be same type
        assert type(ts_panel) == type(gs_panel)
        assert isinstance(gs_panel, PanelLike)

    def test_panellike_validator_works(self):
        """Test that TimeSmith validators work with GeoSmith PanelLike."""
        panel_data = pd.DataFrame(
            {
                "entity": ["A", "A", "B", "B"],
                "value": np.random.randn(4),
            },
            index=pd.date_range("2020-01-01", periods=4, freq="D"),
        )

        gs_panel = GSPanelLike(data=panel_data, entity_col="entity")

        # TimeSmith validator should work
        assert_panel_like(gs_panel)

    def test_tablelike_is_same_type(self):
        """Test that GeoSmith TableLike is same as TimeSmith TableLike."""
        table_data = pd.DataFrame(
            {
                "feature1": np.random.randn(10),
                "feature2": np.random.randn(10),
            },
            index=pd.date_range("2020-01-01", periods=10, freq="D"),
        )

        ts_table = TableLike(data=table_data)
        gs_table = GSTableLike(data=table_data)

        # Should be same type
        assert type(ts_table) == type(gs_table)
        assert isinstance(gs_table, TableLike)

    def test_tablelike_validator_works(self):
        """Test that TimeSmith validators work with GeoSmith TableLike."""
        table_data = pd.DataFrame(
            {
                "feature1": np.random.randn(10),
                "feature2": np.random.randn(10),
            },
            index=pd.date_range("2020-01-01", periods=10, freq="D"),
        )

        gs_table = GSTableLike(data=table_data)

        # TimeSmith validator should work
        assert_table_like(gs_table)

    def test_no_circular_imports(self):
        """Test that there are no circular imports."""
        # This test passes if imports succeed
        from geosmith import PointSet, SeriesLike, PanelLike, TableLike

        assert PointSet is not None
        assert SeriesLike is not None
        assert PanelLike is not None
        assert TableLike is not None

