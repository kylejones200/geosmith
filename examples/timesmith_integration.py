"""Integration example: GeoSmith with TimeSmith typing.

This example demonstrates GeoSmith's integration with TimeSmith's typing layer,
proving that shared types work across the *Smith ecosystem.

Requirements:
- timesmith must be installed
- This example validates using timesmith.typing.validators
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

# Import GeoSmith objects (which also use timesmith.typing)
from geosmith import PointSet
from geosmith.objects._timesmith_compat import (
    validate_panel_like,
    validate_series_like,
    validate_table_like,
)


def main():
    """Run TimeSmith integration example."""
    print("=" * 60)
    print("GeoSmith + TimeSmith Integration Example")
    print("=" * 60)

    # Create a pandas Series with time index
    print("\n1. Creating time series data...")
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    values = np.random.randn(100).cumsum() + 100
    series_data = pd.Series(values, index=dates, name="temperature")

    # Validate using timesmith.typing.validators
    print("\n2. Validating with timesmith.typing.validators...")
    series_like = SeriesLike(data=series_data, name="temperature")
    assert_series_like(series_like)
    print(f"   ✓ SeriesLike validated: {series_like}")

    # Use GeoSmith's validation helper
    validated = validate_series_like(series_data)
    print(f"   ✓ Validated via GeoSmith helper: {validated}")

    # Create panel data
    print("\n3. Creating panel data...")
    panel_data = pd.DataFrame(
        {
            "entity": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "value": np.random.randn(9),
        },
        index=pd.date_range("2020-01-01", periods=9, freq="D"),
    )

    panel_like = PanelLike(data=panel_data, entity_col="entity")
    assert_panel_like(panel_like)
    print(f"   ✓ PanelLike validated: {panel_like}")

    # Create table data
    print("\n4. Creating table data...")
    table_data = pd.DataFrame(
        {
            "feature1": np.random.randn(50),
            "feature2": np.random.randn(50),
        },
        index=pd.date_range("2020-01-01", periods=50, freq="D"),
    )

    table_like = TableLike(data=table_data)
    assert_table_like(table_like)
    print(f"   ✓ TableLike validated: {table_like}")

    # Demonstrate GeoSmith spatial objects work independently
    print("\n5. GeoSmith spatial objects work independently...")
    coords = np.random.rand(10, 2) * 100
    points = PointSet(coordinates=coords)
    print(f"   ✓ PointSet created: {len(points.coordinates)} points")

    print("\n" + "=" * 60)
    print("Integration test passed!")
    print("=" * 60)
    print("\nKey points:")
    print("- SeriesLike, PanelLike, TableLike imported from timesmith.typing")
    print("- Validators from timesmith.typing.validators work correctly")
    print("- GeoSmith spatial objects work independently")
    print("- No circular imports, clean dependency graph")


if __name__ == "__main__":
    main()


