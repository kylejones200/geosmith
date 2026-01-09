"""Layer 3: Tasks - User intent translation.

Tasks translate user intent into object creation, primitive calls, and model runs.
Tasks must not import matplotlib. Tasks can import geopandas and rasterio if
present, but keep these imports optional and isolated.
"""

from geosmith.tasks.blockmodeltask import BlockModelTask
from geosmith.tasks.changetask import ChangeTask
from geosmith.tasks.crossvalidation import SpatialKFold, WellBasedKFold
from geosmith.tasks.faciestask import FaciesResult, FaciesTask
from geosmith.tasks.featuretask import FeatureTask
from geosmith.tasks.rastertask import RasterTask
from geosmith.tasks.routetask import RouteTask

__all__ = [
    "BlockModelTask",
    "ChangeTask",
    "FaciesResult",
    "FaciesTask",
    "FeatureTask",
    "RasterTask",
    "RouteTask",
    "SpatialKFold",
    "WellBasedKFold",
]

