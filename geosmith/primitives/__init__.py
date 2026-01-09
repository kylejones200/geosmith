"""Layer 2: Primitives - Algorithm interfaces and pure operations.

This layer defines algorithm interfaces and pure operations. It can import
numpy and pandas. It can optionally import shapely and pyproj behind optional
dependencies with clean fallbacks. No file I/O or plotting.
"""

from geosmith.primitives.base import (
    BaseEstimator,
    BaseObject,
    BaseRasterModel,
    BaseSpatialModel,
    BaseTransformer,
)
from geosmith.primitives.geometry import (
    bounding_box,
    line_length,
    nearest_neighbor_search,
    point_in_polygon,
    polygon_area,
)
from geosmith.primitives.crs import (
    get_common_crs,
    get_epsg_code,
    standardize_crs,
    transform_coordinates,
    validate_coordinates,
)
from geosmith.primitives.geomechanics import (
    calculate_effective_stress,
    calculate_hydrostatic_pressure,
    calculate_overburden_stress,
    calculate_overpressure,
    calculate_pressure_gradient,
    calculate_stress_ratio,
    drucker_prager_failure,
    estimate_shmin_from_poisson,
    hoek_brown_failure,
    mohr_coulomb_failure,
    pressure_to_mud_weight,
    stress_polygon_limits,
)
from geosmith.primitives.interpolation import idw_interpolate, idw_to_raster

# Optional kriging (requires scipy)
try:
    from geosmith.primitives.kriging import KrigingResult, OrdinaryKriging

    KRIGING_AVAILABLE = True
except ImportError:
    KRIGING_AVAILABLE = False
    KrigingResult = None  # type: ignore
    OrdinaryKriging = None  # type: ignore

from geosmith.primitives.petrophysics import (
    ArchieParams,
    calculate_avo_attributes,
    calculate_avo_from_slowness,
    calculate_bulk_volume_water,
    calculate_density_from_velocity,
    calculate_fluid_bulk_modulus,
    calculate_formation_factor,
    calculate_permeability_coates_dumanoir,
    calculate_permeability_kozeny_carman,
    calculate_permeability_porosity_only,
    calculate_permeability_timur,
    calculate_permeability_tixier,
    calculate_permeability_wyllie_rose,
    calculate_porosity_from_density,
    calculate_velocities_from_slowness,
    calculate_water_saturation,
    calculate_water_saturation_indonesia,
    calculate_water_saturation_simandoux,
    calculate_water_saturation_waxman_smits,
    gassmann_fluid_substitution,
    pickett_isolines,
    preprocess_avo_inputs,
)
from geosmith.primitives.raster import grid_resample, zonal_reduce

# Optional feature engineering (requires scikit-learn)
try:
    from geosmith.primitives.features import (
        build_block_model_features,
        build_spatial_features,
    )

    FEATURES_AVAILABLE = True
except ImportError:
    FEATURES_AVAILABLE = False
    build_spatial_features = None  # type: ignore
    build_block_model_features = None  # type: ignore
from geosmith.primitives.simulation import (
    compute_exceedance_probability,
    compute_simulation_statistics,
    sequential_gaussian_simulation,
)
from geosmith.primitives.variogram import (
    VariogramModel,
    compute_experimental_variogram,
    fit_variogram_model,
    predict_variogram,
)

__all__ = [
    "ArchieParams",
    "BaseEstimator",
    "BaseObject",
    "BaseRasterModel",
    "BaseSpatialModel",
    "BaseTransformer",
    "bounding_box",
    "build_block_model_features",
    "build_spatial_features",
    "calculate_avo_attributes",
    "calculate_avo_from_slowness",
    "calculate_bulk_volume_water",
    "calculate_density_from_velocity",
    "calculate_effective_stress",
    "calculate_fluid_bulk_modulus",
    "calculate_formation_factor",
    "calculate_hydrostatic_pressure",
    "calculate_overburden_stress",
    "calculate_overpressure",
    "calculate_permeability_coates_dumanoir",
    "calculate_permeability_kozeny_carman",
    "calculate_permeability_porosity_only",
    "calculate_permeability_timur",
    "calculate_permeability_tixier",
    "calculate_permeability_wyllie_rose",
    "calculate_porosity_from_density",
    "calculate_pressure_gradient",
    "calculate_stress_ratio",
    "calculate_velocities_from_slowness",
    "calculate_water_saturation",
    "calculate_water_saturation_indonesia",
    "calculate_water_saturation_simandoux",
    "calculate_water_saturation_waxman_smits",
    "drucker_prager_failure",
    "estimate_shmin_from_poisson",
    "gassmann_fluid_substitution",
    "hoek_brown_failure",
    "mohr_coulomb_failure",
    "compute_exceedance_probability",
    "compute_experimental_variogram",
    "compute_simulation_statistics",
    "fit_variogram_model",
    "get_common_crs",
    "get_epsg_code",
    "grid_resample",
    "idw_interpolate",
    "idw_to_raster",
    "line_length",
    "nearest_neighbor_search",
    "pickett_isolines",
    "point_in_polygon",
    "polygon_area",
    "preprocess_avo_inputs",
    "pressure_to_mud_weight",
    "predict_variogram",
    "sequential_gaussian_simulation",
    "standardize_crs",
    "stress_polygon_limits",
    "transform_coordinates",
    "validate_coordinates",
    "VariogramModel",
    "zonal_reduce",
]

# Conditionally add kriging exports if available
if KRIGING_AVAILABLE:
    __all__.extend(["KrigingResult", "OrdinaryKriging"])

# Conditionally add feature engineering exports if available
if FEATURES_AVAILABLE:
    # Already in __all__, but we can add them conditionally if needed
    pass

