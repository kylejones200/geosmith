"""Geosmith mining primitives (modular package).

Ore modeling and forecasting operations split into logical modules:
- ore_modeling: Hybrid IDW+ML ore grade estimation
- forecasting: Ore grade forecasting (Kriging, GPR, XGBoost)

Note: Block model operations are in tasks/blockmodeltask.py
      Variogram/kriging/simulation are in primitives/variogram.py, kriging.py, simulation.py
      Feature engineering is in primitives/features.py

This package maintains backward compatibility with the original flat import:
`from geosmith.primitives.mining import ...`
"""

# Import order matters - avoid circular imports

# Forecasting
from geosmith.primitives.mining.forecasting import (
    analyze_uncertainty_calibration,
    compare_forecasting_methods,
    create_prediction_grid,
    create_spatial_folds,
    fit_variogram_for_forecasting,
    generate_synthetic_geochemical_data,
    ordinary_kriging_predict,
    prepare_spatial_features,
    train_gaussian_process,
    train_xgboost,
)

# Ore Modeling
from geosmith.primitives.mining.ore_modeling import (
    HybridModelResults,
    HybridOreModel,
    predict_block_grades,
    train_hybrid_model,
)

__all__ = ['HybridModelResults', 'HybridOreModel', 'analyze_uncertainty_calibration', 'compare_forecasting_methods', 'create_prediction_grid', 'create_spatial_folds', 'fit_variogram_for_forecasting', 'generate_synthetic_geochemical_data', 'ordinary_kriging_predict', 'predict_block_grades', 'prepare_spatial_features', 'train_gaussian_process', 'train_hybrid_model', 'train_xgboost']
