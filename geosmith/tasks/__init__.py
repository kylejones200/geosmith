"""Layer 3: Tasks - User intent translation.

Tasks translate user intent into object creation, primitive calls, and model runs.
Tasks must not import matplotlib. Tasks can import geopandas and rasterio if
present, but keep these imports optional and isolated.
"""

from geosmith.utils.optional_imports import optional_import
from geosmith.tasks.blockmodeltask import BlockModelTask, create_block_model_grid
from geosmith.tasks.changetask import ChangeTask
from geosmith.tasks.featuretask import FeatureTask
from geosmith.tasks.rastertask import RasterTask
from geosmith.tasks.routetask import RouteTask

# Optional stratigraphy (requires ruptures and scipy)
STRATIGRAPHY_AVAILABLE, _strat = optional_import(
    "geosmith.tasks.stratigraphy",
    [
        "ChangePointResult",
        "StratigraphyTask",
        "detect_bayesian_online",
        "detect_pelt",
        "preprocess_log",
    ],
)
ChangePointResult = _strat["ChangePointResult"]  # type: ignore
StratigraphyTask = _strat["StratigraphyTask"]  # type: ignore
detect_bayesian_online = _strat["detect_bayesian_online"]  # type: ignore
detect_pelt = _strat["detect_pelt"]  # type: ignore
preprocess_log = _strat["preprocess_log"]  # type: ignore

# Optional surrogate models (requires scikit-learn, xgboost)
SURROGATE_AVAILABLE, _surrogate = optional_import(
    "geosmith.tasks.surrogatetask", ["SurrogateTask"]
)
SurrogateTask = _surrogate["SurrogateTask"]  # type: ignore

# Optional NLP (requires spacy or transformers)
NLP_AVAILABLE, _nlp = optional_import(
    "geosmith.tasks.nlptask",
    ["ChronostratNER", "EntityMatch", "NERResult", "extract_chronostrat_entities"],
)
ChronostratNER = _nlp["ChronostratNER"]  # type: ignore
EntityMatch = _nlp["EntityMatch"]  # type: ignore
NERResult = _nlp["NERResult"]  # type: ignore
extract_chronostrat_entities = _nlp["extract_chronostrat_entities"]  # type: ignore

# Optional facies classification (requires scikit-learn)
FACIES_AVAILABLE, _facies = optional_import(
    "geosmith.tasks.faciestask", ["FaciesResult", "FaciesTask"]
)
FaciesResult = _facies["FaciesResult"]  # type: ignore
FaciesTask = _facies["FaciesTask"]  # type: ignore

# Optional clustering (requires scikit-learn)
CLUSTERING_AVAILABLE, _clustering = optional_import(
    "geosmith.tasks.clusteringtask",
    ["FaciesClusterer", "cluster_facies", "find_optimal_clusters"],
)
FaciesClusterer = _clustering["FaciesClusterer"]  # type: ignore
cluster_facies = _clustering["cluster_facies"]  # type: ignore
find_optimal_clusters = _clustering["find_optimal_clusters"]  # type: ignore

# Optional regression (requires scikit-learn)
REGRESSION_AVAILABLE, _regression = optional_import(
    "geosmith.tasks.regressiontask", ["PermeabilityPredictor", "PorosityPredictor"]
)
PermeabilityPredictor = _regression["PermeabilityPredictor"]  # type: ignore
PorosityPredictor = _regression["PorosityPredictor"]  # type: ignore

# Optional decline curve analysis (requires scipy)
DECLINE_AVAILABLE, _decline = optional_import(
    "geosmith.tasks.declinetask",
    [
        "DeclineModel",
        "ExponentialDecline",
        "HarmonicDecline",
        "HyperbolicDecline",
        "fit_decline_model",
        "forecast_production",
        "process_wells_parallel",
    ],
)
DeclineModel = _decline["DeclineModel"]  # type: ignore
ExponentialDecline = _decline["ExponentialDecline"]  # type: ignore
HarmonicDecline = _decline["HarmonicDecline"]  # type: ignore
HyperbolicDecline = _decline["HyperbolicDecline"]  # type: ignore
fit_decline_model = _decline["fit_decline_model"]  # type: ignore
forecast_production = _decline["forecast_production"]  # type: ignore
process_wells_parallel = _decline["process_wells_parallel"]  # type: ignore

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
