"""Layer 3: Tasks - User intent translation.

Tasks translate user intent into object creation, primitive calls, and model runs.
Tasks must not import matplotlib. Tasks can import geopandas and rasterio if
present, but keep these imports optional and isolated.
"""

from geosmith.tasks.blockmodeltask import BlockModelTask, create_block_model_grid
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

# Optional surrogate models (requires scikit-learn, xgboost)
try:
    from geosmith.tasks.surrogatetask import SurrogateTask

    SURROGATE_AVAILABLE = True
except ImportError:
    SURROGATE_AVAILABLE = False
    SurrogateTask = None  # type: ignore

# Optional NLP (requires spacy or transformers)
try:
    from geosmith.tasks.nlptask import (
        ChronostratNER,
        EntityMatch,
        NERResult,
        extract_chronostrat_entities,
    )

    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
    ChronostratNER = None  # type: ignore
    EntityMatch = None  # type: ignore
    NERResult = None  # type: ignore
    extract_chronostrat_entities = None  # type: ignore

# Optional facies classification (requires scikit-learn)
try:
    from geosmith.tasks.faciestask import FaciesResult, FaciesTask

    FACIES_AVAILABLE = True
except ImportError:
    FACIES_AVAILABLE = False
    FaciesResult = None  # type: ignore
    FaciesTask = None  # type: ignore

# Optional clustering (requires scikit-learn)
try:
    from geosmith.tasks.clusteringtask import (
        FaciesClusterer,
        cluster_facies,
        find_optimal_clusters,
    )

    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False
    FaciesClusterer = None  # type: ignore
    cluster_facies = None  # type: ignore
    find_optimal_clusters = None  # type: ignore

# Optional regression (requires scikit-learn)
try:
    from geosmith.tasks.regressiontask import PermeabilityPredictor, PorosityPredictor

    REGRESSION_AVAILABLE = True
except ImportError:
    REGRESSION_AVAILABLE = False
    PermeabilityPredictor = None  # type: ignore
    PorosityPredictor = None  # type: ignore

# Optional decline curve analysis (requires scipy)
try:
    from geosmith.tasks.declinetask import (
        DeclineModel,
        ExponentialDecline,
        HarmonicDecline,
        HyperbolicDecline,
        fit_decline_model,
        forecast_production,
        process_wells_parallel,
    )

    DECLINE_AVAILABLE = True
except ImportError:
    DECLINE_AVAILABLE = False
    DeclineModel = None  # type: ignore
    ExponentialDecline = None  # type: ignore
    HarmonicDecline = None  # type: ignore
    HyperbolicDecline = None  # type: ignore
    fit_decline_model = None  # type: ignore
    forecast_production = None  # type: ignore
    process_wells_parallel = None  # type: ignore

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
    "create_block_model_grid",
    "FeatureTask",
    "RasterTask",
    "RouteTask",
    "StratigraphyTask",
]

# Conditionally add ML exports if available
if FACIES_AVAILABLE:
    __all__.extend(["FaciesResult", "FaciesTask"])
if CLUSTERING_AVAILABLE:
    __all__.extend(["FaciesClusterer", "cluster_facies", "find_optimal_clusters"])
if REGRESSION_AVAILABLE:
    __all__.extend(["PermeabilityPredictor", "PorosityPredictor"])
if DECLINE_AVAILABLE:
    __all__.extend(
        [
            "DeclineModel",
            "ExponentialDecline",
            "HarmonicDecline",
            "HyperbolicDecline",
            "fit_decline_model",
            "forecast_production",
            "process_wells_parallel",
        ]
    )
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
if SURROGATE_AVAILABLE:
    __all__.extend(["SurrogateTask"])
if NLP_AVAILABLE:
    __all__.extend(
        [
            "ChronostratNER",
            "EntityMatch",
            "NERResult",
            "extract_chronostrat_entities",
        ]
    )

