"""Geosmith ML primitives (modular package).

Pure ML operations split into logical modules:
- interpretability: Model interpretability calculations (feature importance, SHAP)
- cross_validation: Cross-validation schemes for geoscience data
- model_utils: Model evaluation utilities (confusion matrix calculations)
- hyperparameter: Hyperparameter optimization utilities

This package maintains backward compatibility with the original flat import:
`from geosmith.primitives.ml import ...`
"""

# Import order matters - avoid circular imports

# Cross Validation
from geosmith.primitives.ml.cross_validation import (
    SpatialCrossValidator,
    WellBasedKFold,
)

# Hyperparameter
from geosmith.primitives.ml.hyperparameter import (
    SubsurfaceHyperparameterOptimizer,
    objective,
    optimize_facies_classifier,
    optimize_property_predictor,
)

# Interpretability (pure calculations only, no plotting)
from geosmith.primitives.ml.interpretability import (
    calculate_shap_values,
    get_feature_importance,
)

# Model Utils
from geosmith.primitives.ml.model_utils import (
    _adjust_confusion_matrix_kernel,
    compute_metrics_from_cm,
    confusion_matrix_to_dataframe,
    display_adj_cm,
    display_cm,
)

__all__ = [
    "SpatialCrossValidator",
    "SubsurfaceHyperparameterOptimizer",
    "WellBasedKFold",
    "_adjust_confusion_matrix_kernel",
    "calculate_shap_values",
    "compute_metrics_from_cm",
    "confusion_matrix_to_dataframe",
    "display_adj_cm",
    "display_cm",
    "get_feature_importance",
    "objective",
    "optimize_facies_classifier",
    "optimize_property_predictor",
    "partial_dependence_plot",
]
