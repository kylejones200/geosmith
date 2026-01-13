"""Spatial cross-validation task.

Migrated from geosuite.ml.cross_validation.
Layer 3: Tasks - User intent translation.

Re-exports cross-validation classes from primitives layer for convenience.
This avoids code duplication and ensures consistent behavior across layers.
"""

# Re-export from primitives layer to avoid duplication
try:
    from geosmith.primitives.ml.cross_validation import (
        SpatialCrossValidator,
        WellBasedKFold,
    )

    # Alias SpatialCrossValidator as SpatialKFold for backward compatibility
    SpatialKFold = SpatialCrossValidator

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    WellBasedKFold = None  # type: ignore
    SpatialKFold = None  # type: ignore
    SpatialCrossValidator = None  # type: ignore
