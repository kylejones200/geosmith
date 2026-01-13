"""Geosmith petrophysics (modular package).

Pure petrophysics operations split into logical modules:
- water_saturation: Water saturation calculations (Archie, Simandoux,
  Indonesia, Waxman-Smits)
- permeability: Permeability calculations (Kozeny-Carman, Timur, Tixier, etc.)
- rock_physics: Rock physics calculations (Gassmann, fluid properties, porosity)
- avo: AVO (Amplitude vs Offset) attribute calculations
- plots: Plotting utilities (Pickett plot isolines)

This package maintains backward compatibility with the original flat import:
`from geosmith.primitives.petrophysics import ...`
"""

# Import order matters - avoid circular imports

# Water saturation (no dependencies on other petrophysics modules)
from geosmith.primitives.petrophysics.water_saturation import (
    ArchieParams,
    calculate_bulk_volume_water,
    calculate_water_saturation,
    calculate_water_saturation_indonesia,
    calculate_water_saturation_simandoux,
    calculate_water_saturation_waxman_smits,
)

# Permeability (independent)
from geosmith.primitives.petrophysics.permeability import (
    calculate_permeability_coates_dumanoir,
    calculate_permeability_kozeny_carman,
    calculate_permeability_porosity_only,
    calculate_permeability_timur,
    calculate_permeability_tixier,
    calculate_permeability_wyllie_rose,
)

# Rock physics (independent)
from geosmith.primitives.petrophysics.rock_physics import (
    calculate_density_from_velocity,
    calculate_fluid_bulk_modulus,
    calculate_formation_factor,
    calculate_porosity_from_density,
    calculate_velocities_from_slowness,
    gassmann_fluid_substitution,
)

# AVO (independent)
from geosmith.primitives.petrophysics.avo import (
    calculate_avo_attributes,
    calculate_avo_from_slowness,
    preprocess_avo_inputs,
)

# Plots (may depend on water_saturation for Pickett)
from geosmith.primitives.petrophysics.plots import (
    pickett_isolines,
)

# Private kernel functions (exported for internal use, but typically not for users)
from geosmith.primitives.petrophysics.plots import (
    _pickett_isolines_kernel,
)

__all__ = [
    # Water saturation
    "ArchieParams",
    "calculate_bulk_volume_water",
    "calculate_water_saturation",
    "calculate_water_saturation_indonesia",
    "calculate_water_saturation_simandoux",
    "calculate_water_saturation_waxman_smits",
    # Permeability
    "calculate_permeability_coates_dumanoir",
    "calculate_permeability_kozeny_carman",
    "calculate_permeability_porosity_only",
    "calculate_permeability_timur",
    "calculate_permeability_tixier",
    "calculate_permeability_wyllie_rose",
    # Rock physics
    "calculate_density_from_velocity",
    "calculate_fluid_bulk_modulus",
    "calculate_formation_factor",
    "calculate_porosity_from_density",
    "calculate_velocities_from_slowness",
    "gassmann_fluid_substitution",
    # AVO
    "calculate_avo_attributes",
    "calculate_avo_from_slowness",
    "preprocess_avo_inputs",
    # Plots
    "pickett_isolines",
    # Private kernels
    "_pickett_isolines_kernel",
]
