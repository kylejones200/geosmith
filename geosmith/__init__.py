"""GeoSmith: *Smith for ML models with strict 4-layer architecture.

Public API - only workflows and base types are exposed.
"""

# Layer 1: Objects
from geosmith.objects import (
    AnomalyScores,
    GeoIndex,
    GeoTable,
    LineSet,
    PanelLike,
    PointSet,
    PolygonSet,
    RasterGrid,
    SeriesLike,
    SpatialAnomalyResult,
    TableLike,
)

# Layer 2: Base classes
from geosmith.primitives.base import (
    BaseEstimator,
    BaseObject,
    BaseRasterModel,
    BaseSpatialModel,
    BaseTransformer,
)

# Layer 3: Tasks (exposed for advanced usage)
from geosmith.tasks import BlockModelTask

# Layer 4: Workflows (public API)
from geosmith.workflows import (
    GeostatisticalModel,
    GeostatisticalResult,
    make_features,
    process_raster,
    read_grdecl,
    read_raster,
    read_vector,
    reproject_to,
    write_grdecl,
    write_raster,
    write_vector,
    zonal_stats,
)

__version__ = "0.1.0"

__all__ = [
    # Objects
    "AnomalyScores",
    "GeoIndex",
    "GeoTable",
    "LineSet",
    "PanelLike",
    "PointSet",
    "PolygonSet",
    "RasterGrid",
    "SeriesLike",
    "SpatialAnomalyResult",
    "TableLike",
    # Base classes
    "BaseEstimator",
    "BaseObject",
    "BaseRasterModel",
    "BaseSpatialModel",
    "BaseTransformer",
    # Tasks
    "BlockModelTask",
    # Workflows
    "GeostatisticalModel",
    "GeostatisticalResult",
    "make_features",
    "process_raster",
    "read_grdecl",
    "read_raster",
    "read_vector",
    "reproject_to",
    "write_grdecl",
    "write_raster",
    "write_vector",
    "zonal_stats",
]
