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

# Geomechanics is now a modular package - import from package __init__
from geosmith.primitives.geomechanics import (
    # Stress
    calculate_effective_stress,
    calculate_stress_ratio,
    determine_stress_regime,
    estimate_shmin_from_poisson,
    friction_coefficient_ratio,
    shmax_bounds,
    stress_polygon_limits,
    stress_polygon_points,
    wellbore_stress_concentration,
    # Pressure
    calculate_hydrostatic_pressure,
    calculate_overburden_stress,
    calculate_overpressure,
    calculate_pore_pressure_bowers,
    calculate_pore_pressure_eaton_sonic,
    calculate_pressure_gradient,
    mud_weight_equivalent,
    pore_pressure_eaton,
    pressure_to_mud_weight,
    sv_from_density,
    # Private kernels (exported for internal use)
    _calculate_overburden_stress_kernel,
    _calculate_pressure_gradient_kernel,
    # Failure
    drucker_prager_failure,
    hoek_brown_failure,
    mohr_coulomb_failure,
    # Fracture
    calculate_fracture_aperture,
    calculate_fracture_permeability,
    fracture_orientation_distribution,
    predict_fracture_orientation,
    # Inversion
    invert_stress_combined,
    invert_stress_from_breakout,
    invert_stress_from_dif,
    # Parallel
    calculate_overburden_stress_parallel,
    get_parallel_info,
    process_well_array_parallel,
    # Wellbore (dataclasses and functions)
    GeomechParams,
    WellboreStabilityResult,
    breakout_analysis,
    drilling_margin_analysis,
    safe_mud_weight_window,
    wellbore_stress_plot_data,
    # Field
    FieldOptimizationResult,
    calculate_field_statistics,
    process_field_data,
)
from geosmith.primitives.interpolation import (
    compute_idw_residuals,
    idw_interpolate,
    idw_to_raster,
)

# Optional kriging (requires scipy)
try:
    from geosmith.primitives.kriging import KrigingResult, OrdinaryKriging

    KRIGING_AVAILABLE = True
except ImportError:
    KRIGING_AVAILABLE = False
    KrigingResult = None  # type: ignore
    OrdinaryKriging = None  # type: ignore

# Petrophysics imports (required, not optional)
try:
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
except ImportError as e:
    # If petrophysics fails to import, this is a critical error
    # but we'll raise it with a clearer message
    raise ImportError(
        f"Failed to import petrophysics module. This is required. "
        f"Original error: {e}. "
        f"Check that geosmith.primitives.petrophysics.water_saturation can be imported."
    ) from e
from geosmith.primitives.raster import grid_resample, zonal_reduce

# Optional seismic processing (requires scipy)
try:
    from geosmith.primitives.seismic import (
        apply_phase_shift,
        compute_hilbert_attributes,
        correct_trace_phase,
        estimate_residual_phase,
    )

    SEISMIC_AVAILABLE = True
except ImportError:
    SEISMIC_AVAILABLE = False
    apply_phase_shift = None  # type: ignore
    compute_hilbert_attributes = None  # type: ignore
    correct_trace_phase = None  # type: ignore
    estimate_residual_phase = None  # type: ignore

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
    exp_transform,
    log_transform,
    sequential_gaussian_simulation,
)

# Optional surrogate models (requires scikit-learn, xgboost)
try:
    from geosmith.primitives.surrogate import (
        SurrogateModel,
        SurrogateMetrics,
        train_simulation_emulator,
    )

    SURROGATE_AVAILABLE = True
except ImportError:
    SURROGATE_AVAILABLE = False
    SurrogateModel = None  # type: ignore
    SurrogateMetrics = None  # type: ignore
    train_simulation_emulator = None  # type: ignore
from geosmith.primitives.variogram import (
    VariogramModel,
    compute_experimental_variogram,
    fit_variogram_model,
    predict_variogram,
)

# Optional production analysis (requires pandas - usually available)
try:
    from geosmith.primitives.production import (
        aggregate_by_county,
        aggregate_by_field,
        aggregate_by_pool,
        analyze_spatial_distribution,
        analyze_temporal_coverage,
        calculate_production_density,
        calculate_production_summary,
        calculate_well_statistics,
        identify_production_hotspots,
    )

    PRODUCTION_AVAILABLE = True
except ImportError:
    PRODUCTION_AVAILABLE = False
    aggregate_by_county = None  # type: ignore
    aggregate_by_field = None  # type: ignore
    aggregate_by_pool = None  # type: ignore
    analyze_spatial_distribution = None  # type: ignore
    analyze_temporal_coverage = None  # type: ignore
    calculate_production_density = None  # type: ignore
    calculate_production_summary = None  # type: ignore
    calculate_well_statistics = None  # type: ignore
    identify_production_hotspots = None  # type: ignore

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
    "calculate_fracture_aperture",
    "calculate_fracture_permeability",
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
    "fracture_orientation_distribution",
    "friction_coefficient_ratio",
    "gassmann_fluid_substitution",
    "hoek_brown_failure",
    "invert_stress_combined",
    "invert_stress_from_breakout",
    "invert_stress_from_dif",
    "mohr_coulomb_failure",
    "mud_weight_equivalent",
    "compute_exceedance_probability",
    "compute_experimental_variogram",
    "compute_simulation_statistics",
    "exp_transform",
    "fit_variogram_model",
    "log_transform",
    "get_common_crs",
    "compute_idw_residuals",
    "get_epsg_code",
    "grid_resample",
    "idw_interpolate",
    "idw_to_raster",
    "line_length",
    "nearest_neighbor_search",
    "pickett_isolines",
    "point_in_polygon",
    "polygon_area",
    "_calculate_overburden_stress_kernel",
    "_calculate_pressure_gradient_kernel",
    "calculate_overburden_stress_parallel",
    "calculate_pore_pressure_bowers",
    "calculate_pore_pressure_eaton_sonic",
    "determine_stress_regime",
    "get_parallel_info",
    "pore_pressure_eaton",
    "preprocess_avo_inputs",
    "process_well_array_parallel",
    "predict_fracture_orientation",
    "pressure_to_mud_weight",
    "predict_variogram",
    "sequential_gaussian_simulation",
    "shmax_bounds",
    "apply_phase_shift",
    "compute_hilbert_attributes",
    "correct_trace_phase",
    "estimate_residual_phase",
    "standardize_crs",
    "stress_polygon_limits",
    "stress_polygon_points",
    "sv_from_density",
    "transform_coordinates",
    "validate_coordinates",
    "VariogramModel",
    "wellbore_stress_concentration",
    "zonal_reduce",
    # Wellbore stability (added from modular geomechanics)
    "GeomechParams",
    "WellboreStabilityResult",
    "breakout_analysis",
    "drilling_margin_analysis",
    "safe_mud_weight_window",
    "wellbore_stress_plot_data",
    # Field analysis (added from modular geomechanics)
    "FieldOptimizationResult",
    "calculate_field_statistics",
    "process_field_data",
    # Production analysis (added conditionally)
    "aggregate_by_county",
    "aggregate_by_field",
    "aggregate_by_pool",
    "analyze_spatial_distribution",
    "analyze_temporal_coverage",
    "calculate_production_density",
    "calculate_production_summary",
    "calculate_well_statistics",
    "identify_production_hotspots",
]

# Conditionally add kriging exports if available
if KRIGING_AVAILABLE:
    __all__.extend(["KrigingResult", "OrdinaryKriging"])

# Conditionally add feature engineering exports if available
if FEATURES_AVAILABLE:
    # Already in __all__, but we can add them conditionally if needed
    pass

# Conditionally add seismic processing exports if available
if SEISMIC_AVAILABLE:
    # Already in __all__, but we can add them conditionally if needed
    pass

# Conditionally add production exports if available
if PRODUCTION_AVAILABLE:
    # Already in __all__, but keep for clarity
    pass

# Conditionally add surrogate model exports if available
if SURROGATE_AVAILABLE:
    __all__.extend(["SurrogateModel", "SurrogateMetrics", "train_simulation_emulator"])
