#!/usr/bin/env python3
"""Script to automatically refactor geomechanics.py into modular package structure.

Uses AST parsing for accurate extraction of complete function/class definitions.
"""

import ast
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
        "patterns": ["invert_stress", "_invert_stress"],
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
        "exclude": ["wellbore_stress", "breakout_pressure", "fracture_pressure", "safe_mud_weight_window"],
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
            "poisson",
            "friction_coefficient",
            "regime",
            "wellbore_stress_concentration",
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
        "patterns": ["parallel", "get_parallel_info"],
        "name": "parallel.py",
        "description": "Parallel processing utilities",
    },
}


class FunctionExtractor(ast.NodeVisitor):
    """AST visitor to extract function and class definitions."""
    
    def __init__(self):
        self.functions: List[Tuple[str, int, int, str]] = []  # name, start_line, end_line, source
        self.source_lines: List[str] = []
    
    def visit_FunctionDef(self, node):
        """Extract function definitions."""
        # Get function name
        name = node.name
        
        # Skip the fallback njit decorator function (the one defined in try/except)
        if name == "njit" and not node.decorator_list:
            self.generic_visit(node)
            return
        
        # Find start line - include decorators if present
        if node.decorator_list:
            # Start from first decorator
            start_line = node.decorator_list[0].lineno - 1
        else:
            start_line = node.lineno - 1
        
        # Get function body end line
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else node.lineno + 10  # fallback
        
        # Extract source code including decorators
        func_source = "\n".join(self.source_lines[start_line:end_line])
        
        self.functions.append((name, start_line, end_line, func_source))
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        """Extract class definitions."""
        # Find start line - include decorators if present
        if node.decorator_list:
            start_line = node.decorator_list[0].lineno - 1
        else:
            start_line = node.lineno - 1
        
        # Get class body end line
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 10  # fallback
        
        name = node.name
        class_source = "\n".join(self.source_lines[start_line:end_line])
        
        self.functions.append((name, start_line, end_line, class_source))
        self.generic_visit(node)


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
    
    # Skip helper functions (start with _kernel, _predict_, objective, decorator)
    if name.startswith("_kernel") or name.startswith("_predict_") or name in ["objective", "decorator"]:
        return None
    
    # Default fallback to stress
    return "stress"


def get_module_header(module_name: str, description: str) -> str:
    """Generate module header with imports."""
    base_imports = '''"""Geomechanics: {description}

Pure geomechanics operations - {module_name} module.
Migrated from geosuite.geomech.
Layer 2: Primitives - Pure operations.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

from geosmith.primitives.geomechanics._common import (
    NUMBA_AVAILABLE,
    PANDAS_AVAILABLE,
    njit,
    pd,
)

if TYPE_CHECKING:
    import pandas as pd
'''
    
    # Add module-specific imports
    extra_imports = ""
    if module_name == "fracture":
        extra_imports += """
# Optional scipy for von Mises distribution
try:
    from scipy.stats import vonmises
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
"""
    elif module_name == "inversion":
        extra_imports += """
# Optional scipy for optimization
try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
"""
    elif module_name == "wellbore":
        extra_imports += """
from dataclasses import dataclass

from geosmith.primitives.geomechanics.pressure import (
    calculate_hydrostatic_pressure,
    pore_pressure_eaton,
    sv_from_density,
)
from geosmith.primitives.geomechanics.stress import (
    shmax_bounds,
    wellbore_stress_concentration,
)
from geosmith.primitives.petrophysics import calculate_porosity_from_density
"""
    elif module_name == "field":
        extra_imports += """
from dataclasses import dataclass
"""
    elif module_name == "pressure":
        # njit is already imported from _common, but we need to handle kernel functions
        pass
    elif module_name == "parallel":
        extra_imports += """
from geosmith.primitives._numba_helpers import njit, prange, NUMBA_AVAILABLE as NUMBA_HELPER_AVAILABLE
from geosmith.primitives.geomechanics.pressure import sv_from_density
"""
    
    return base_imports.format(module_name=module_name, description=description) + extra_imports + "\n"


def extract_functions_from_source(source: str) -> List[Tuple[str, str]]:
    """Extract functions using AST parsing."""
    # Parse the source code
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"⚠ Syntax error parsing source: {e}")
        return []
    
    # Extract functions
    extractor = FunctionExtractor()
    extractor.source_lines = source.split("\n")
    extractor.visit(tree)
    
    return extractor.functions


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
    source = backup_file.read_text()
    
    print("Parsing source with AST...")
    functions = extract_functions_from_source(source)
    print(f"Found {len(functions)} functions/classes")
    
    # Group functions by module
    module_functions: Dict[str, List[Tuple[str, str]]] = {mod: [] for mod in MODULE_GROUPS.keys()}
    skipped = []
    
    print("\nGrouping functions by module:")
    for name, start_line, end_line, code in functions:
        module = assign_to_module(name)
        if module is None:
            skipped.append(name)
            print(f"  {name:40s} → SKIP (helper/nested function)")
            continue
        module_functions[module].append((name, code))
        print(f"  {name:40s} → {module}.py")
    
    if skipped:
        print(f"\n⚠ Skipped {len(skipped)} helper/nested functions")
    
    # Create module files
    print("\nCreating module files:")
    for module_name, funcs in module_functions.items():
        if not funcs:
            print(f"⚠ Skipping {module_name}.py (no functions assigned)")
            continue
        
        config = MODULE_GROUPS[module_name]
        file_path = output_dir / config["name"]
        
        # Generate content
        content = get_module_header(module_name, config["description"])
        content += "\n"
        
        # Sort functions by name for consistency (can be improved later to preserve original order)
        # But actually, preserve the order from extraction
        for name, code in funcs:
            content += code.rstrip()  # Remove trailing whitespace
            content += "\n\n"  # Add blank line between functions
        
        # Write file
        file_path.write_text(content)
        
        line_count = len(content.split("\n"))
        print(f"✓ Created {file_path.name:20s} with {len(funcs):2d} functions ({line_count:4d} lines)")
    
    print("\n✅ Refactoring complete!")
    print(f"\nNext steps:")
    print(f"1. Review the generated files in {output_dir}")
    print(f"2. Fix any circular import issues")
    print(f"3. Fix missing imports (e.g., njit, scipy)")
    print(f"4. Test imports: python -c 'from geosmith.primitives.geomechanics import *'")
    print(f"5. Update geosmith/primitives/__init__.py if needed")


if __name__ == "__main__":
    main()

