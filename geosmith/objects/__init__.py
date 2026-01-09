"""Layer 1: Objects - Immutable data representations.

This layer contains only data structures. No I/O libraries, no geopandas,
no shapely, no rasterio, no matplotlib. Only standard library + numpy + pandas.

Time series objects (SeriesLike, PanelLike, TableLike) are compatible with
TimeSmith's typing layer for seamless integration without conversion.
TimeSmith is NOT a hard dependency.
"""

from geosmith.objects.anomaly import AnomalyScores, SpatialAnomalyResult
from geosmith.objects.geoindex import GeoIndex
from geosmith.objects.geotable import GeoTable
from geosmith.objects.lineset import LineSet
from geosmith.objects.pointset import PointSet
from geosmith.objects.polygonset import PolygonSet
from geosmith.objects.rastergrid import RasterGrid
from geosmith.objects.timeseries import PanelLike, SeriesLike, TableLike

__all__ = [
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
]

