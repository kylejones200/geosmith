#!/usr/bin/env python3
"""Script to automatically refactor petrophysics.py into modular package structure.

Uses AST parsing for accurate extraction of complete function/class definitions.
"""

import ast
from pathlib import Path
from typing import Dict, List, Tuple


# Module groupings based on function name patterns
MODULE_GROUPS = {
    "water_saturation": {
        "patterns": [
            "water_saturation",
            "ArchieParams",
            "bulk_volume_water",
            "simandoux",
            "indonesia",
            "waxman_smits",
        ],
        "name": "water_saturation.py",
        "description": "Water saturation calculations (Archie, Simandoux, Indonesia, Waxman-Smits)",
    },
    "permeability": {
        "patterns": [
            "permeability",
            "kozeny_carman",
            "timur",
            "porosity_only",
            "wyllie_rose",
            "coates_dumanoir",
            "tixier",
        ],
        "name": "permeability.py",
        "description": "Permeability calculations (Kozeny-Carman, Timur, Tixier, etc.)",
    },
    "rock_physics": {
        "patterns": [
            "gassmann",
            "fluid_bulk_modulus",
            "density_from_velocity",
            "velocities_from_slowness",
            "porosity_from_density",
            "formation_factor",
        ],
        "name": "rock_physics.py",
        "description": "Rock physics calculations (Gassmann, fluid properties, porosity)",
    },
    "avo": {
        "patterns": [
            "avo",
            "preprocess_avo",
            "calculate_avo",
        ],
        "name": "avo.py",
        "description": "AVO (Amplitude vs Offset) attribute calculations",
    },
    "plots": {
        "patterns": [
            "pickett",
        ],
        "name": "plots.py",
        "description": "Plotting utilities (Pickett plot isolines)",
    },
}


class FunctionExtractor(ast.NodeVisitor):
    """AST visitor to extract function and class definitions."""
    
    def __init__(self, source_lines: List[str]):
        self.functions: List[Tuple[str, int, int, str, str]] = []  # name, start_line, end_line, source, type
        self.source_lines = source_lines
    
    def visit_FunctionDef(self, node):
        """Extract function definitions."""
        name = node.name
        
        # Skip nested helper functions that start with underscore (they should remain nested)
        if name.startswith("_") and name != "__post_init__":
            # Check if this is a nested function by seeing if it's inside another function
            parent = getattr(node, 'parent', None)
            if parent and isinstance(parent, ast.FunctionDef):
                # Skip - this is a nested helper function
                self.generic_visit(node)
                return
        
        # Skip the fallback njit decorator function
        if name == "njit" and not node.decorator_list:
            self.generic_visit(node)
            return
        
        # Skip the decorator function itself (fallback)
        if name == "decorator":
            self.generic_visit(node)
            return
        
        # Find start line - include decorators if present
        if node.decorator_list:
            start_line = node.decorator_list[0].lineno - 1
        else:
            start_line = node.lineno - 1
        
        # Get function body end line
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else node.lineno + 10
        
        # Extract source code including decorators
        func_source = "\n".join(self.source_lines[start_line:end_line])
        
        self.functions.append((name, start_line, end_line, func_source, "function"))
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        """Extract class definitions."""
        if node.decorator_list:
            start_line = node.decorator_list[0].lineno - 1
        else:
            start_line = node.lineno - 1
        
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 10
        
        name = node.name
        class_source = "\n".join(self.source_lines[start_line:end_line])
        
        self.functions.append((name, start_line, end_line, class_source, "class"))
        self.generic_visit(node)


def assign_function_to_module(func_name: str, patterns: Dict[str, Dict]) -> str:
    """Assign a function to a module based on name patterns."""
    func_lower = func_name.lower()
    
    for module_name, config in patterns.items():
        module_patterns = config.get("patterns", [])
        exclude = config.get("exclude", [])
        
        # Check excludes first
        if any(exc.lower() in func_lower for exc in exclude):
            continue
        
        # Check if function matches any pattern
        if any(pattern.lower() in func_lower for pattern in module_patterns):
            return module_name
    
    return "common"


def generate_module_header(module_name: str, description: str, domain: str = "petrophysics") -> str:
    """Generate module header."""
    return f'''"""Geosmith {domain}: {description}

Migrated from geosuite.petro.
Layer 2: Primitives - Pure operations.
"""

from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from geosmith.primitives.petrophysics._common import logger, njit

'''


def generate_package_init(module_assignments: Dict[str, List[Tuple]], domain: str = "petrophysics") -> str:
    """Generate __init__.py for package."""
    content = f'''"""Geosmith {domain} (modular package).

Pure petrophysics operations split into logical modules:
- water_saturation: Water saturation calculations (Archie, Simandoux, Indonesia, Waxman-Smits)
- permeability: Permeability calculations (Kozeny-Carman, Timur, Tixier, etc.)
- rock_physics: Rock physics calculations (Gassmann, fluid properties, porosity)
- avo: AVO (Amplitude vs Offset) attribute calculations
- plots: Plotting utilities (Pickett plot isolines)

This package maintains backward compatibility with the original flat import:
`from geosmith.primitives.{domain} import ...`
"""

# Import order matters - avoid circular imports

# Water saturation (no dependencies on other petrophysics modules)
from geosmith.primitives.{domain}.water_saturation import (
    ArchieParams,
    calculate_bulk_volume_water,
    calculate_water_saturation,
    calculate_water_saturation_indonesia,
    calculate_water_saturation_simandoux,
    calculate_water_saturation_waxman_smits,
)

# Permeability (independent)
from geosmith.primitives.{domain}.permeability import (
    calculate_permeability_coates_dumanoir,
    calculate_permeability_kozeny_carman,
    calculate_permeability_porosity_only,
    calculate_permeability_timur,
    calculate_permeability_tixier,
    calculate_permeability_wyllie_rose,
)

# Rock physics (independent)
from geosmith.primitives.{domain}.rock_physics import (
    calculate_density_from_velocity,
    calculate_fluid_bulk_modulus,
    calculate_formation_factor,
    calculate_porosity_from_density,
    calculate_velocities_from_slowness,
    gassmann_fluid_substitution,
)

# AVO (independent)
from geosmith.primitives.{domain}.avo import (
    calculate_avo_attributes,
    calculate_avo_from_slowness,
    preprocess_avo_inputs,
)

# Plots (may depend on water_saturation for Pickett)
from geosmith.primitives.{domain}.plots import (
    pickett_isolines,
)

# Private kernel functions (exported for internal use, but typically not for users)
from geosmith.primitives.{domain}.plots import (
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
'''
    
    return content


def main():
    """Main execution."""
    backup_file = Path("geosmith/primitives/petrophysics.py.backup")
    source_file = Path("geosmith/primitives/petrophysics.py")
    
    if not backup_file.exists() and source_file.exists():
        # Create backup
        import shutil
        shutil.copy(source_file, backup_file)
        print(f"Created backup: {backup_file}")
    
    if not backup_file.exists():
        print(f"Error: Backup file not found: {backup_file}")
        return
    
    # Read source file
    source_lines = backup_file.read_text().splitlines()
    
    # Parse AST
    tree = ast.parse("\n".join(source_lines))
    
    # Extract functions
    extractor = FunctionExtractor(source_lines)
    
    # Track parent nodes for nested function detection
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            if hasattr(child, 'parent'):
                continue
            child.parent = node
    
    extractor.visit(tree)
    
    print(f"Found {len(extractor.functions)} top-level functions/classes")
    
    # Assign functions to modules
    module_assignments: Dict[str, List[Tuple]] = {name: [] for name in MODULE_GROUPS.keys()}
    module_assignments["common"] = []
    
    for func_name, start_line, end_line, func_source, func_type in extractor.functions:
        # Skip nested helpers that should remain in their parent functions
        if func_name.startswith("_") and func_type == "function":
            # Check if it's a kernel helper (these should be extracted separately or kept nested)
            if "_kernel" in func_name or func_name == "_pickett_isolines_kernel":
                # Keep these for now - they might need special handling
                module_name = assign_function_to_module(func_name, MODULE_GROUPS)
                module_assignments[module_name].append((func_name, start_line, end_line, func_source, func_type))
            continue
        
        module_name = assign_function_to_module(func_name, MODULE_GROUPS)
        module_assignments[module_name].append((func_name, start_line, end_line, func_source, func_type))
    
    # Create package directory
    package_dir = Path("geosmith/primitives/petrophysics")
    package_dir.mkdir(exist_ok=True)
    
    # Create common utilities file first
    common_file = package_dir / "_common.py"
    common_content = '''"""Common utilities and constants for petrophysics modules."""

import logging

logger = logging.getLogger(__name__)

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])
'''
    common_file.write_text(common_content)
    
    # Generate modules
    for module_name, funcs in module_assignments.items():
        if module_name == "common" and not funcs:
            continue
        
        if not funcs:
            print(f"⚠ No functions assigned to {module_name}")
            continue
        
        module_config = MODULE_GROUPS.get(module_name, {"description": module_name.replace("_", " ").title(), "name": f"{module_name}.py"})
        
        module_file = package_dir / module_config["name"]
        header = generate_module_header(module_name, module_config["description"])
        
        # Sort functions by line number to maintain order
        funcs_sorted = sorted(funcs, key=lambda x: x[1])
        
        # Combine function sources
        module_content = header + "\n\n\n".join(
            func_source for _, _, _, func_source, _ in funcs_sorted
        ) + "\n"
        
        module_file.write_text(module_content)
        print(f"✅ Created {module_file}: {len(funcs)} functions/classes")
    
    # Generate __init__.py
    init_file = package_dir / "__init__.py"
    init_content = generate_package_init(module_assignments)
    init_file.write_text(init_content)
    print(f"✅ Created {init_file}")
    
    # Summary
    print(f"\n📦 Refactored petrophysics.py into package:")
    for module_name, funcs in module_assignments.items():
        if funcs:
            print(f"   {module_name}: {len(funcs)} functions/classes")


if __name__ == "__main__":
    main()

