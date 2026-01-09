"""Example: GeoSmith and TimeSmith integration.

This example demonstrates GeoSmith's integration with TimeSmith's typing layer.
TimeSmith is a required dependency for time series functionality.

All time series types (SeriesLike, PanelLike, TableLike) are imported from
timesmith.typing (single source of truth) for ecosystem compatibility.
"""

import numpy as np
import pandas as pd

# Import from timesmith.typing (single source of truth)
from timesmith.typing import PanelLike, SeriesLike, TableLike
from timesmith.typing.validators import (
    assert_panel_like,
    assert_series_like,
    assert_table_like,
)

from geosmith import PointSet

# Set random seed for reproducibility
np.random.seed(42)

print("GeoSmith and TimeSmith Integration Example")
print("=" * 60)

# Create a time series
dates = pd.date_range("2020-01-01", periods=100, freq="D")
values = np.random.randn(100).cumsum() + 100
series_data = pd.Series(values, index=dates, name="temperature")

# Create SeriesLike (pandas Series conforms to SeriesLike Protocol)
series_like: SeriesLike = series_data
assert_series_like(series_like)
print(f"\n1. Created SeriesLike (pandas Series conforms to Protocol): {series_like}")
print(f"   ✓ Validated using timesmith.typing.validators.assert_series_like")

# Create panel data
panel_data = pd.DataFrame(
    {
        "entity": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        "value": np.random.randn(9),
    },
    index=pd.date_range("2020-01-01", periods=9, freq="D"),
)

panel_like: PanelLike = panel_data
assert_panel_like(panel_like)
print(f"\n2. Created PanelLike (pandas DataFrame conforms to Protocol): {panel_like}")
print(f"   ✓ Validated using timesmith.typing.validators.assert_panel_like")

# Create table data
table_data = pd.DataFrame(
    {
        "feature1": np.random.randn(50),
        "feature2": np.random.randn(50),
        "feature3": np.random.randn(50),
    },
    index=pd.date_range("2020-01-01", periods=50, freq="D"),
)

table_like: TableLike = table_data
assert_table_like(table_like)
print(f"\n3. Created TableLike (pandas DataFrame conforms to Protocol): {table_like}")
print(f"   ✓ Validated using timesmith.typing.validators.assert_table_like")

# Demonstrate GeoSmith spatial objects work independently
print("\n4. GeoSmith spatial objects work independently...")
coords = np.random.rand(10, 2) * 100
points = PointSet(coordinates=coords)
print(f"   ✓ PointSet created: {len(points.coordinates)} points")

print("\n" + "=" * 60)
print("Integration example completed!")
print("=" * 60)
print("\nKey points:")
print("- SeriesLike, PanelLike, TableLike imported from timesmith.typing")
print("- Validators from timesmith.typing.validators work correctly")
print("- GeoSmith spatial objects work independently")
print("- No circular imports, clean dependency graph")
print("- timesmith is a required dependency (not optional)")
