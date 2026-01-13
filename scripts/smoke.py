#!/usr/bin/env python3
"""Smoke test for geosmith integration with timesmith.

This script validates that geosmith works correctly with timesmith typing when needed.
It must:
1. Check if timesmith is available (optional dependency)
2. If available, validate SeriesLike via timesmith.typing
3. Exit 0 on success
"""

import sys

import numpy as np
import pandas as pd

try:
    # Import from timesmith.typing (single source of truth)
    from timesmith.typing import SeriesLike
    from timesmith.typing.validators import assert_series_like

    TIMESMITH_AVAILABLE = True
except ImportError:
    TIMESMITH_AVAILABLE = False
    print("ℹ timesmith not available - skipping timesmith integration test")


def main() -> int:
    """Run smoke test."""
    if not TIMESMITH_AVAILABLE:
        print("✓ Smoke test passed: geosmith works without timesmith (optional dependency)")
        return 0

    # Create a pandas Series
    np.random.seed(42)
    n = 50
    values = np.random.randn(n) * 2 + 10
    index = pd.date_range("2020-01-01", periods=n, freq="D")
    y: SeriesLike = pd.Series(values, index=index, name="Test Series")

    # Validate via timesmith.typing
    try:
        assert_series_like(y)
        print("✓ Validated SeriesLike via timesmith.typing")
    except Exception as e:
        print(f"ERROR: timesmith validation failed: {e}", file=sys.stderr)
        return 1

    # Import geosmith time series objects
    try:
        from geosmith.objects.timeseries import (
            assert_series_like as geo_assert_series_like,
        )

        # Verify they work together
        geo_assert_series_like(y)
        print("✓ geosmith timeseries objects work with timesmith.typing")

        print("✓ Smoke test passed: geosmith works with timesmith typing")
        return 0

    except Exception as e:
        print(f"ERROR: geosmith integration failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

