#!/usr/bin/env python3
"""Script to split plotting.py into modular structure.

This extracts functions from the monolithic plotting.py file and organizes
them into logical modules within the plotting/ package.
"""

import re
from pathlib import Path

# Read the original file
plotting_file = Path("geosmith/workflows/plotting.py")
with plotting_file.open("r") as f:
    lines = f.readlines()

# Function categories with their line ranges (1-indexed, inclusive)
categories = {
    "petrophysics": {
        "start": 40,
        "end": 405,
        "functions": ["buckles_plot", "pickett_plot", "neutron_density_crossplot"],
    },
    "geomechanics": {
        "start": 406,
        "end": 577,
        "functions": ["plot_pressure_profile", "plot_mud_weight_profile"],
    },
    "well_logs": {
        "start": 585,  # Includes helper functions _is_plotsmith_available, _apply_plotsmith_style
        "end": 1053,
        "functions": [
            "create_strip_chart",
            "add_log_track",
            "add_facies_track",
            "create_facies_log_plot",
            "create_multi_well_strip_chart",
        ],
    },
    "lithology": {
        "start": 1054,
        "end": 1584,
        "functions": [
            "ternary_plot",
            "sand_silt_clay_plot",
            "qfl_plot",
            "mineral_composition_plot",
            "_ternary_to_cartesian",
            "_cartesian_to_ternary",
            "_setup_ternary_axes",
        ],
    },
    "interactive": {
        "start": 1585,
        "end": 2110,
        "functions": [
            "create_interactive_kriging_map",
            "create_interactive_well_map",
            "create_combined_map",
        ],
    },
    "maps": {
        "start": 2111,
        "end": 2555,
        "functions": [
            "create_field_map",
            "create_well_trajectory_map",
            "_create_folium_map",
            "_create_geopandas_map",
            "_create_3d_trajectory_map",
            "_create_2d_trajectory_map",
        ],
    },
    "_3d": {
        "start": 2556,
        "end": 2988,
        "functions": [
            "create_3d_well_log_viewer",
            "create_multi_well_3d_viewer",
            "create_cross_section_viewer",
        ],
    },
}

# Extract header (lines 1-38: imports and setup)
header_lines = lines[0:38]
header = "".join(header_lines)

# Extract helper functions and constants (lines 585-672)
helpers_start = 585
helpers_end = 672
helpers_lines = lines[helpers_start - 1 : helpers_end]
helpers_section = "".join(helpers_lines)

print("✅ Analysis complete")
print(f"Header: {len(header_lines)} lines")
print(f"Helpers: {len(helpers_lines)} lines")
for cat, info in categories.items():
    cat_lines = lines[info["start"] - 1 : info["end"]]
    print(f"{cat}: {len(cat_lines)} lines")

