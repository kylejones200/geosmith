"""Layer 3: Tasks - User intent translation.

Tasks translate user intent into object creation, primitive calls, and model runs.
Tasks must not import matplotlib. Tasks can import geopandas and rasterio if
present, but keep these imports optional and isolated.
"""

from geosmith.tasks.blockmodeltask import BlockModelTask
from geosmith.tasks.changetask import ChangeTask
from geosmith.tasks.featuretask import FeatureTask
from geosmith.tasks.rastertask import RasterTask

# Optional stratigraphy (requires ruptures and scipy)
try:
    from geosmith.tasks.stratigraphy import (
        ChangePointResult,
        StratigraphyTask,
        detect_bayesian_online,
        detect_pelt,
        preprocess_log,
    )

    STRATIGRAPHY_AVAILABLE = True
except ImportError:
    STRATIGRAPHY_AVAILABLE = False
    ChangePointResult = None  # type: ignore
    StratigraphyTask = None  # type: ignore
    detect_bayesian_online = None  # type: ignore
    detect_pelt = None  # type: ignore
    preprocess_log = None  # type: ignore
from geosmith.tasks.routetask import RouteTask

# Optional facies classification (requires scikit-learn)
try:
    from geosmith.tasks.faciestask import FaciesResult, FaciesTask

    FACIES_AVAILABLE = True
except ImportError:
    FACIES_AVAILABLE = False
    FaciesResult = None  # type: ignore
    FaciesTask = None  # type: ignore

# Optional cross-validation (requires scikit-learn)
try:
    from geosmith.tasks.crossvalidation import SpatialKFold, WellBasedKFold

    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False
    SpatialKFold = None  # type: ignore
    WellBasedKFold = None  # type: ignore

__all__ = [
    "BlockModelTask",
    "ChangeTask",
    "FeatureTask",
    "RasterTask",
    "RouteTask",
    "StratigraphyTask",
]

# Conditionally add ML exports if available
if FACIES_AVAILABLE:
    __all__.extend(["FaciesResult", "FaciesTask"])
if CV_AVAILABLE:
    __all__.extend(["SpatialKFold", "WellBasedKFold"])
if STRATIGRAPHY_AVAILABLE:
    __all__.extend(
        [
            "ChangePointResult",
            "StratigraphyTask",
            "detect_bayesian_online",
            "detect_pelt",
            "preprocess_log",
        ]
    )

