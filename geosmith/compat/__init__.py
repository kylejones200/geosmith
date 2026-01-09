"""Compatibility layer for pygeomodeling deprecation.

This module provides compatibility shims for pygeomodeling code,
allowing gradual migration to GeoSmith's 4-layer architecture.
"""

# Import compatibility shims
try:
    from geosmith.compat.pygeomodeling import (
        GRDECLParser,
        OrdinaryKriging as PyGeoOrdinaryKriging,
        UnifiedSPE9Toolkit,
        VariogramModel as PyGeoVariogramModel,
        compute_experimental_variogram as pygeo_compute_variogram,
        fit_variogram_model as pygeo_fit_variogram,
    )

    __all__ = [
        "GRDECLParser",
        "OrdinaryKriging",
        "PyGeoOrdinaryKriging",
        "UnifiedSPE9Toolkit",
        "VariogramModel",
        "PyGeoVariogramModel",
        "compute_experimental_variogram",
        "pygeo_compute_variogram",
        "fit_variogram_model",
        "pygeo_fit_variogram",
    ]

    # Aliases for backward compatibility
    OrdinaryKriging = PyGeoOrdinaryKriging
    VariogramModel = PyGeoVariogramModel
    compute_experimental_variogram = pygeo_compute_variogram
    fit_variogram_model = pygeo_fit_variogram

except ImportError:
    __all__ = []

