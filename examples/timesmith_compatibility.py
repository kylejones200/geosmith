"""Example: GeoSmith and TimeSmith compatibility.

This example demonstrates how GeoSmith's time series objects are compatible
with TimeSmith's typing layer, allowing seamless integration without conversion.

TimeSmith is NOT a hard dependency - GeoSmith works independently.
"""

import pandas as pd
import numpy as np

from geosmith import SeriesLike, PanelLike, TableLike

# Set random seed for reproducibility
np.random.seed(42)

print("GeoSmith and TimeSmith Compatibility Example")
print("=" * 60)

# Create a time series
dates = pd.date_range("2020-01-01", periods=100, freq="D")
values = np.random.randn(100).cumsum() + 100
series_data = pd.Series(values, index=dates, name="temperature")

# Create GeoSmith SeriesLike
gs_series = SeriesLike(data=series_data, name="temperature")
print(f"\n1. Created GeoSmith SeriesLike: {gs_series}")

# Try to convert to TimeSmith (if available)
try:
    ts_series = gs_series.to_timesmith()
    print(f"   ✓ Converted to TimeSmith SeriesLike (TimeSmith is available)")
    print(f"   Type: {type(ts_series)}")
except ImportError:
    print(f"   ℹ TimeSmith not available - using GeoSmith SeriesLike")
    print(f"   Install with: pip install geosmith[timesmith]")

# Create panel data
panel_data = pd.DataFrame(
    {
        "entity": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        "value": np.random.randn(9),
    },
    index=pd.date_range("2020-01-01", periods=9, freq="D"),
)

gs_panel = PanelLike(data=panel_data, entity_col="entity")
print(f"\n2. Created GeoSmith PanelLike: {gs_panel}")

# Try to convert to TimeSmith (if available)
try:
    ts_panel = gs_panel.to_timesmith()
    print(f"   ✓ Converted to TimeSmith PanelLike (TimeSmith is available)")
    print(f"   Type: {type(ts_panel)}")
except ImportError:
    print(f"   ℹ TimeSmith not available - using GeoSmith PanelLike")

# Create table data
table_data = pd.DataFrame(
    {
        "feature1": np.random.randn(50),
        "feature2": np.random.randn(50),
        "feature3": np.random.randn(50),
    },
    index=pd.date_range("2020-01-01", periods=50, freq="D"),
)

gs_table = TableLike(data=table_data)
print(f"\n3. Created GeoSmith TableLike: {gs_table}")

# Try to convert to TimeSmith (if available)
try:
    ts_table = gs_table.to_timesmith()
    print(f"   ✓ Converted to TimeSmith TableLike (TimeSmith is available)")
    print(f"   Type: {type(ts_table)}")
except ImportError:
    print(f"   ℹ TimeSmith not available - using GeoSmith TableLike")

print("\n" + "=" * 60)
print("Summary:")
print("- GeoSmith objects work independently (no TimeSmith required)")
print("- When TimeSmith is available, objects can be converted seamlessly")
print("- No data conversion needed - same underlying pandas structures")
print("- Install TimeSmith: pip install geosmith[timesmith]")

