#!/usr/bin/env python3
"""Script to automatically refactor geomechanics.py into modular package structure.

This script parses the backup file and extracts functions/classes into logical modules.
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Tuple


# Module groupings based on function name patterns
MODULE_GROUPS = {
    "failure": {
        "patterns": ["mohr_coulomb", "drucker_prager", "hoek_brown"],
        "name": "failure.py",
        "description": "Failure criteria calculations",
    },
    "fracture": {
        "patterns": [
            "fracture_orientation",
            "predict_fracture",
            "fracture_aperture",
            "fracture_permeability",
            "_predict_coulomb",
            "_predict_griffith",
            "_predict_tensile",
        ],
        "name": "fracture.py",
        "description": "Fracture orientation and permeability",
    },
    "inversion": {
        "patterns": [
            "invert_stress",
            "_invert_stress",
        ],
        "name": "inversion.py",
        "description": "Stress inversion from observations",
    },
    "pressure": {
        "patterns": [
            "pressure",
            "pore_pressure",
            "hydrostatic",
            "overpressure",
            "mud_weight",
            "sv_from_density",
            "overburden",
            "_calculate_pressure",
            "_calculate_overburden",
        ],
        "exclude": ["wellbore_stress", "breakout_pressure", "fracture_pressure"],
        "name": "pressure.py",
        "description": "Pressure calculations",
    },
    "stress": {
        "patterns": [
            "stress_ratio",
            "stress_polygon",
            "shmax",
            "shmin",
            "effective_stress",
            "overpressure",  # This might be in pressure, but also relates to stress
            "poisson",
            "friction_coefficient",
            "regime",
            "wellbore_stress_concentration",  # Stress concentration around wellbore
        ],
        "exclude": ["wellbore_stress_plot", "breakout", "fracture"],
        "name": "stress.py",
        "description": "Basic stress calculations",
    },
    "wellbore": {
        "patterns": [
            "wellbore",
            "breakout",
            "GeomechParams",
            "WellboreStabilityResult",
            "safe_mud_weight_window",
            "drilling_margin",
        ],
        "name": "wellbore.py",
        "description": "Wellbore stability analysis",
    },
    "field": {
        "patterns": [
            "FieldOptimizationResult",
            "process_field_data",
            "calculate_field_statistics",
            "optimize_field",
            "field_correlation",
            "drilling_trends",
            "field_recommendations",
            "economic_analysis",
            "risk_assessment",
            "field_development",
        ],
        "name": "field.py",
        "description": "Field-wide analysis",
    },
    "parallel": {
        "patterns": [
            "parallel",
            "get_parallel_info",
        ],
        "name": "parallel.py",
        "description": "Parallel processing utilities",
    },
}


def read_backup_file(backup_path: Path) -> str:
    """Read the backup geomechanics file."""
    with open(backup_path, "r") as f:
        return f.read()


def find_function_ranges(content: str) -> List[Tuple[str, int, int]]:
    """Find all function/class definitions and their line ranges.
    
    Returns list of (name, start_line, end_line) tuples.
    """
    lines = content.split("\n")
    functions = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Check for function or class definition
        if line.startswith("def ") or line.startswith("class ") or line.startswith("@dataclass"):
            # Extract name
            if line.startswith("@dataclass"):
                # Next line should be class definition
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line.startswith("class "):
                        name = next_line.split("(")[0].replace("class ", "").strip()
                        start = i
                        # Find the end of the class
                        i += 1
                        indent_level = len(next_line) - len(next_line.lstrip())
                        while i + 1 < len(lines):
                            i += 1
                            current_line = lines[i].strip()
                            if not current_line:
                                continue
                            current_indent = len(lines[i]) - len(lines[i].lstrip())
                            # If we hit a line at same or lower indent (and not just whitespace), we're done
                            if current_indent <= indent_level and current_line:
                                break
                        functions.append((name, start, i))
                        continue
            else:
                # Regular function or class
                name = line.split("(")[0].split(":")[0].replace("def ", "").replace("class ", "").strip()
                start = i
                # Find the end of the function/class
                indent_level = len(lines[i]) - len(lines[i].lstrip())
                in_docstring = False
                while i + 1 < len(lines):
                    i += 1
                    current_line = lines[i]
                    current_stripped = current_line.strip()
                    current_indent = len(current_line) - len(current_line.lstrip())
                    
                    # Track docstrings
                    if '"""' in current_line or "'''" in current_line:
                        in_docstring = not in_docstring
                        continue
                    
                    # If we hit a line at same or lower indent (and not empty/comment/docstring continuation), we're done
                    if not in_docstring and current_stripped and not current_stripped.startswith("#"):
                        if current_indent <= indent_level:
                            break
                
                functions.append((name, start, i))
                continue
        i += 1
    
    return functions


def assign_to_module(name: str) -> str:
    """Assign a function/class to a module based on its name."""
    name_lower = name.lower()
    
    for module_name, config in MODULE_GROUPS.items():
        # Check patterns
        matches = any(pattern.lower() in name_lower for pattern in config["patterns"])
        
        if matches:
            # Check exclusions
            if "exclude" in config:
                excluded = any(excl.lower() in name_lower for excl in config["exclude"])
                if excluded:
                    continue
            return module_name
    
    # Default: assign to stress if it's a stress-related function
    if "stress" in name_lower or "shmax" in name_lower or "shmin" in name_lower:
        return "stress"
    
    return "stress"  # Default fallback


def extract_function_code(content: str, start_line: int, end_line: int) -> str:
    """Extract function/class code from content."""
    lines = content.split("\n")
    return "\n".join(lines[start_line:end_line])


def get_module_header(module_name: str, description: str) -> str:
    """Generate module header with imports."""
    common_imports = """from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

from geosmith.primitives.geomechanics._common import (
    NUMBA_AVAILABLE,
    PANDAS_AVAILABLE,
    njit,
    pd,
)

if TYPE_CHECKING:
    import pandas as pd
"""
    
    # Add module-specific imports
    if module_name == "failure":
        pass  # No special imports
    elif module_name == "fracture":
        # Fracture might need scipy for von Mises
        common_imports += """
# Optional scipy for von Mises distribution
try:
    from scipy.stats import vonmises
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
"""
    elif module_name == "inversion":
        common_imports += """
# Optional scipy for optimization
try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
"""
    elif module_name == "wellbore":
        common_imports += """
from dataclasses import dataclass

from geosmith.primitives.geomechanics.pressure import (
    calculate_hydrostatic_pressure,
    pore_pressure_eaton,
    sv_from_density,
)
from geosmith.primitives.geomechanics.stress import shmax_bounds, wellbore_stress_concentration
"""
    elif module_name == "field":
        common_imports += """
from dataclasses import dataclass
"""
    elif module_name == "pressure":
        common_imports += """
from geosmith.primitives._numba_helpers import njit as _njit
"""
    
    return f'''"""Geomechanics: {description}

Pure geomechanics operations - {module_name} module.
Migrated from geosuite.geomech.
Layer 2: Primitives - Pure operations.
"""

{common_imports}
'''


def create_module_file(
    module_name: str,
    functions: List[Tuple[str, str]],
    output_dir: Path,
) -> None:
    """Create a module file with extracted functions."""
    config = MODULE_GROUPS[module_name]
    file_path = output_dir / config["name"]
    
    # Generate header
    content = get_module_header(module_name, config["description"])
    content += "\n\n"
    
    # Add functions in order
    for name, code in functions:
        content += code
        content += "\n\n"
    
    # Write file
    with open(file_path, "w") as f:
        f.write(content)
    
    print(f"✓ Created {file_path} with {len(functions)} functions")


def main():
    """Main refactoring function."""
    base_dir = Path(__file__).parent.parent
    backup_file = base_dir / "geosmith/primitives/geomechanics.py.backup"
    output_dir = base_dir / "geosmith/primitives/geomechanics"
    
    if not backup_file.exists():
        print(f"❌ Backup file not found: {backup_file}")
        return
    
    output_dir.mkdir(exist_ok=True)
    
    print("Reading backup file...")
    content = read_backup_file(backup_file)
    
    print("Finding function definitions...")
    functions = find_function_ranges(content)
    print(f"Found {len(functions)} functions/classes")
    
    # Group functions by module
    module_functions: Dict[str, List[Tuple[str, str]]] = {mod: [] for mod in MODULE_GROUPS.keys()}
    
    print("\nGrouping functions by module:")
    for name, start, end in functions:
        module = assign_to_module(name)
        code = extract_function_code(content, start, end)
        module_functions[module].append((name, code))
        print(f"  {name:40s} → {module}.py")
    
    # Create module files
    print("\nCreating module files:")
    for module_name, funcs in module_functions.items():
        if funcs:
            create_module_file(module_name, funcs, output_dir)
        else:
            print(f"⚠ Skipping {module_name}.py (no functions assigned)")
    
    print("\n✅ Refactoring complete!")
    print(f"\nNext steps:")
    print(f"1. Review the generated files in {output_dir}")
    print(f"2. Fix any import issues (cross-module dependencies)")
    print(f"3. Test imports: python -c 'from geosmith.primitives.geomechanics import *'")
    print(f"4. Update geosmith/primitives/__init__.py if needed")


if __name__ == "__main__":
    main()

