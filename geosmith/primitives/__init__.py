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
    calculate_overpressure,
    calculate_pressure_gradient,
    calculate_stress_ratio,
    pressure_to_mud_weight,
)
from geosmith.primitives.interpolation import idw_interpolate, idw_to_raster
from geosmith.primitives.kriging import KrigingResult, OrdinaryKriging
from geosmith.primitives.petrophysics import (
    ArchieParams,
    calculate_bulk_volume_water,
    calculate_water_saturation,
    pickett_isolines,
)
from geosmith.primitives.raster import grid_resample, zonal_reduce
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
    "calculate_bulk_volume_water",
    "calculate_effective_stress",
    "calculate_overpressure",
    "calculate_pressure_gradient",
    "calculate_stress_ratio",
    "calculate_water_saturation",
    "compute_exceedance_probability",
    "compute_experimental_variogram",
    "compute_simulation_statistics",
    "fit_variogram_model",
    "get_common_crs",
    "get_epsg_code",
    "grid_resample",
    "idw_interpolate",
    "idw_to_raster",
    "KrigingResult",
    "line_length",
    "nearest_neighbor_search",
    "OrdinaryKriging",
    "pickett_isolines",
    "point_in_polygon",
    "polygon_area",
    "pressure_to_mud_weight",
    "predict_variogram",
    "sequential_gaussian_simulation",
    "standardize_crs",
    "transform_coordinates",
    "validate_coordinates",
    "VariogramModel",
    "zonal_reduce",
]

