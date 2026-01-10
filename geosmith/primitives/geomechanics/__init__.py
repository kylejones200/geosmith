"""Geomechanics calculation primitives (modular package).

Pure geomechanics operations split into logical modules:
- stress: Basic stress calculations
- pressure: Pressure calculations  
- failure: Failure criteria
- fracture: Fracture orientation and permeability
- inversion: Stress inversion from observations
- wellbore: Wellbore stability analysis
- field: Field-wide analysis
- parallel: Parallel processing utilities

This package maintains backward compatibility with the original flat import:
`from geosmith.primitives.geomechanics import ...`
"""

# Import order matters - avoid circular imports by importing base modules first
# Then modules that depend on them

# Stress calculations (no dependencies on other geomechanics modules)
from geosmith.primitives.geomechanics.stress import (
    calculate_effective_stress,
    calculate_stress_ratio,
    determine_stress_regime,
    estimate_shmin_from_poisson,
    friction_coefficient_ratio,
    shmax_bounds,
    stress_polygon_limits,
    stress_polygon_points,
    wellbore_stress_concentration,
)

# Pressure calculations (may use stress for some functions, but mostly independent)
from geosmith.primitives.geomechanics.pressure import (
    _calculate_overburden_stress_kernel,
    _calculate_pressure_gradient_kernel,
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
)

# Failure criteria (independent)
from geosmith.primitives.geomechanics.failure import (
    drucker_prager_failure,
    hoek_brown_failure,
    mohr_coulomb_failure,
)

# Fracture orientation (independent)
from geosmith.primitives.geomechanics.fracture import (
    _predict_coulomb_fracture,
    _predict_griffith_fracture,
    _predict_tensile_fracture,
    calculate_fracture_aperture,
    calculate_fracture_permeability,
    fracture_orientation_distribution,
    predict_fracture_orientation,
)

# Stress inversion (may use stress calculations)
from geosmith.primitives.geomechanics.inversion import (
    _invert_stress_analytical,
    _invert_stress_optimization,
    invert_stress_combined,
    invert_stress_from_breakout,
    invert_stress_from_dif,
)

# Parallel processing (depends on pressure for sv_from_density)
from geosmith.primitives.geomechanics.parallel import (
    calculate_overburden_stress_parallel,
    get_parallel_info,
    process_well_array_parallel,
)

# Wellbore stability (depends on pressure, stress, and petrophysics)
# Import last to avoid circular imports
from geosmith.primitives.geomechanics.wellbore import (
    GeomechParams,
    WellboreStabilityResult,
    breakout_analysis,
    drilling_margin_analysis,
    safe_mud_weight_window,
    wellbore_stress_plot_data,
)

# Field analysis (depends on wellbore)
from geosmith.primitives.geomechanics.field import (
    FieldOptimizationResult,
    calculate_field_statistics,
    process_field_data,
)

# Private kernel functions - typically not exported, but available if needed
# They're already imported above from their respective modules

__all__ = [
    # Stress
    "calculate_effective_stress",
    "calculate_stress_ratio",
    "determine_stress_regime",
    "estimate_shmin_from_poisson",
    "friction_coefficient_ratio",
    "shmax_bounds",
    "stress_polygon_limits",
    "stress_polygon_points",
    "wellbore_stress_concentration",
    # Pressure
    "calculate_hydrostatic_pressure",
    "calculate_overburden_stress",
    "calculate_overpressure",
    "calculate_pore_pressure_bowers",
    "calculate_pore_pressure_eaton_sonic",
    "calculate_pressure_gradient",
    "mud_weight_equivalent",
    "pore_pressure_eaton",
    "pressure_to_mud_weight",
    "sv_from_density",
    # Private kernels (exported for internal use, but typically not for users)
    "_calculate_overburden_stress_kernel",
    "_calculate_pressure_gradient_kernel",
    # Failure
    "drucker_prager_failure",
    "hoek_brown_failure",
    "mohr_coulomb_failure",
    # Fracture
    "calculate_fracture_aperture",
    "calculate_fracture_permeability",
    "fracture_orientation_distribution",
    "predict_fracture_orientation",
    # Private fracture helpers (exported for internal use)
    "_predict_coulomb_fracture",
    "_predict_griffith_fracture",
    "_predict_tensile_fracture",
    # Inversion
    "invert_stress_combined",
    "invert_stress_from_breakout",
    "invert_stress_from_dif",
    # Private inversion helpers (exported for internal use)
    "_invert_stress_analytical",
    "_invert_stress_optimization",
    # Parallel
    "calculate_overburden_stress_parallel",
    "get_parallel_info",
    "process_well_array_parallel",
    # Wellbore
    "GeomechParams",
    "WellboreStabilityResult",
    "breakout_analysis",
    "drilling_margin_analysis",
    "safe_mud_weight_window",
    "wellbore_stress_plot_data",
    # Field
    "FieldOptimizationResult",
    "calculate_field_statistics",
    "process_field_data",
]
