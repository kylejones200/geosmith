#!/usr/bin/env python3
"""Batch Migration Tool - Fast migration while maintaining 4-layer architecture.

Usage:
    python scripts/batch_migrate_module.py geosuite/io geosmith/workflows/io --layer workflows --domain io
"""

import ast
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Layer constraints
LAYER_RULES = {
    "objects": {
        "allowed_imports": ["numpy", "pandas", "typing", "dataclasses"],
        "description": "Immutable data representations",
    },
    "primitives": {
        "allowed_imports": ["numpy", "pandas", "shapely", "pyproj", "numba", "scipy", "sklearn"],
        "description": "Pure algorithm operations",
        "forbidden": ["matplotlib", "rasterio", "geopandas", "xarray"],  # Only optional adapters
    },
    "tasks": {
        "allowed_imports": ["numpy", "pandas", "geopandas", "rasterio", "xarray"],
        "description": "User intent translation",
        "forbidden": ["matplotlib"],  # No plotting in tasks
    },
    "workflows": {
        "allowed_imports": ["numpy", "pandas", "geopandas", "rasterio", "xarray", "matplotlib", "plotsmith", "timesmith"],
        "description": "Public API, I/O, plotting",
    },
}

# Max file size per domain (maintains modularity)
MAX_FILE_SIZES = {
    "petrophysics": 600,
    "geomechanics": 600,
    "production": 500,
    "geometry": 500,
    "io": 700,  # I/O can be larger due to format handling
    "plotting": 700,
    "mining": 600,
    "ml": 500,
    "utils": 400,
}


def extract_functions_from_file(source_file: Path) -> List[Dict]:
    """Extract all functions and classes from a Python file using AST."""
    try:
        content = source_file.read_text()
        tree = ast.parse(content)
        
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                start_line = node.lineno - 1
                end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 50
                
                # Include decorators
                if node.decorator_list:
                    start_line = node.decorator_list[0].lineno - 1
                
                func_code = "\n".join(content.split("\n")[start_line:end_line])
                functions.append({
                    "name": node.name,
                    "type": "function",
                    "code": func_code,
                    "start_line": start_line,
                    "end_line": end_line,
                })
            elif isinstance(node, ast.ClassDef):
                start_line = node.lineno - 1
                end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 100
                
                if node.decorator_list:
                    start_line = node.decorator_list[0].lineno - 1
                
                class_code = "\n".join(content.split("\n")[start_line:end_line])
                functions.append({
                    "name": node.name,
                    "type": "class",
                    "code": class_code,
                    "start_line": start_line,
                    "end_line": end_line,
                })
        
        return functions
    except Exception as e:
        print(f"⚠ Error parsing {source_file}: {e}")
        return []


def check_layer_compliance(code: str, target_layer: str) -> List[str]:
    """Check if code complies with layer rules."""
    violations = []
    rules = LAYER_RULES.get(target_layer, {})
    forbidden = rules.get("forbidden", [])
    
    # Check imports
    for forbidden_import in forbidden:
        if re.search(rf"^\s*import\s+{forbidden_import}|^\s*from\s+{forbidden_import}", code, re.MULTILINE):
            violations.append(f"Forbidden import: {forbidden_import} in {target_layer} layer")
    
    # Check for file I/O in primitives
    if target_layer == "primitives":
        if re.search(r"open\s*\(|\.read\s*\(|\.write\s*\(", code):
            violations.append("File I/O detected in primitives layer (should be in workflows)")
    
    # Check for plotting in tasks
    if target_layer == "tasks":
        if re.search(r"plt\.|matplotlib|plot\s*\(", code, re.IGNORECASE):
            violations.append("Plotting detected in tasks layer (should be in workflows)")
    
    return violations


def adapt_imports(code: str, source_module: str, target_module: str, target_layer: str) -> str:
    """Adapt imports for target layer and module."""
    # Replace source module imports
    code = re.sub(
        rf"from\s+{re.escape(source_module)}\.",
        f"# Migrated from {source_module}\n",
        code
    )
    
    # Handle optional dependencies
    if target_layer == "primitives":
        # Wrap heavy imports in try/except
        for heavy_dep in ["scipy", "sklearn", "skgstat", "pykrige"]:
            pattern = rf"^\s*import\s+{heavy_dep}"
            replacement = f"try:\n    import {heavy_dep}\n    {heavy_dep.upper()}_AVAILABLE = True\nexcept ImportError:\n    {heavy_dep.upper()}_AVAILABLE = False"
            code = re.sub(pattern, replacement, code, flags=re.MULTILINE)
    
    return code


def batch_migrate_module(
    source_dir: Path,
    target_dir: Path,
    target_layer: str,
    domain: str,
    max_file_size: int = 600,
) -> Dict[str, any]:
    """Migrate entire module while maintaining architecture."""
    
    if not source_dir.exists():
        return {"error": f"Source directory not found: {source_dir}"}
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all Python files from source
    source_files = list(source_dir.rglob("*.py"))
    source_files = [f for f in source_files if "__pycache__" not in str(f) and "test" not in f.name.lower()]
    
    all_functions = []
    violations = []
    
    # Extract all functions
    for source_file in source_files:
        functions = extract_functions_from_file(source_file)
        for func in functions:
            # Check compliance
            func_violations = check_layer_compliance(func["code"], target_layer)
            if func_violations:
                violations.extend(func_violations)
            
            # Adapt imports
            func["code"] = adapt_imports(
                func["code"],
                str(source_dir),
                str(target_dir),
                target_layer
            )
        
        all_functions.extend(functions)
    
    # Check if we need to split into package
    total_lines = sum(len(func["code"].split("\n")) for func in all_functions)
    needs_splitting = total_lines > max_file_size
    
    if needs_splitting:
        # Group by logical subdomain
        grouped = group_functions_by_domain(all_functions, domain)
        
        # Create package structure
        package_dir = target_dir / domain
        package_dir.mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py
        init_content = generate_package_init(grouped, domain, target_layer)
        (package_dir / "__init__.py").write_text(init_content)
        
        # Create submodules
        for submodule, funcs in grouped.items():
            module_file = package_dir / f"{submodule}.py"
            module_content = generate_module_content(funcs, domain, submodule, target_layer)
            module_file.write_text(module_content)
        
        result = {
            "migrated": len(all_functions),
            "split_into": list(grouped.keys()),
            "target": str(package_dir),
            "violations": violations,
        }
    else:
        # Single file migration
        target_file = target_dir / f"{domain}.py"
        file_content = generate_single_file_content(all_functions, domain, target_layer)
        target_file.write_text(file_content)
        
        result = {
            "migrated": len(all_functions),
            "target": str(target_file),
            "violations": violations,
        }
    
    return result


def group_functions_by_domain(functions: List[Dict], domain: str) -> Dict[str, List[Dict]]:
    """Group functions by logical subdomain."""
    groups = defaultdict(list)
    
    # Domain-specific grouping patterns
    patterns = {
        "petrophysics": {
            "water_saturation": ["water_saturation", "sw", "saturation", "archie"],
            "permeability": ["permeability", "k_", "kozeny", "timur", "tixier"],
            "rock_physics": ["gassmann", "avo", "velocity", "density", "bulk_modulus"],
            "lithology": ["lithology", "mineral", "ternary", "qfl"],
            "plots": ["plot", "pickett", "buckles", "crossplot"],
        },
        "io": {
            "vector_io": ["read_vector", "write_vector", "geopandas", "shapefile"],
            "raster_io": ["read_raster", "write_raster", "rasterio", "tiff"],
            "witsml": ["witsml", "xml"],
            "dlis": ["dlis"],
            "resqml": ["resqml"],
            "ppdm": ["ppdm"],
            "csv": ["csv", "read_csv"],
            "las": ["las", "read_las"],
        },
        "geospatial": {
            "distance": ["distance", "haversine", "euclidean"],
            "neighbors": ["neighbor", "knn", "k_nearest"],
            "polygons": ["polygon", "intersect", "union", "buffer"],
            "transforms": ["transform", "reproject", "crs"],
        },
    }
    
    domain_patterns = patterns.get(domain, {})
    
    for func in functions:
        name_lower = func["name"].lower()
        assigned = False
        
        for group, keywords in domain_patterns.items():
            if any(keyword in name_lower for keyword in keywords):
                groups[group].append(func)
                assigned = True
                break
        
        if not assigned:
            groups["common"].append(func)
    
    return dict(groups)


def generate_module_content(functions: List[Dict], domain: str, submodule: str, layer: str) -> str:
    """Generate content for a submodule."""
    header = f'''"""Geosmith {domain}: {submodule.replace('_', ' ').title()}

Migrated from geosuite.{domain}.
Layer {get_layer_number(layer)}: {LAYER_RULES[layer]["description"]}.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

'''
    
    # Add domain-specific imports
    if layer == "primitives":
        header += '''
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None  # type: ignore

'''
    
    content = header + "\n\n".join(func["code"] for func in functions) + "\n"
    return content


def generate_single_file_content(functions: List[Dict], domain: str, layer: str) -> str:
    """Generate content for single file migration."""
    return generate_module_content(functions, domain, domain, layer)


def generate_package_init(groups: Dict[str, List[Dict]], domain: str, layer: str) -> str:
    """Generate __init__.py for package."""
    content = f'''"""Geosmith {domain} (modular package).

Migrated from geosuite.{domain}.
Layer {get_layer_number(layer)}: {LAYER_RULES[layer]["description"]}.
"""

'''
    
    # Import from submodules
    for submodule, funcs in groups.items():
        if submodule == "common":
            continue
        for func in funcs:
            content += f"from geosmith.{layer}.{domain}.{submodule} import {func['name']}\n"
    
    # Common imports
    if "common" in groups:
        for func in groups["common"]:
            content += f"from geosmith.{layer}.{domain}.common import {func['name']}\n"
    
    # Generate __all__
    all_names = [func["name"] for funcs in groups.values() for func in funcs]
    content += f"\n__all__ = {all_names!r}\n"
    
    return content


def get_layer_number(layer: str) -> int:
    """Get layer number from name."""
    mapping = {"objects": 1, "primitives": 2, "tasks": 3, "workflows": 4}
    return mapping.get(layer, 2)


def main():
    """CLI entry point."""
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python batch_migrate_module.py <source_dir> <target_dir> <layer> [domain]")
        print("Example: python batch_migrate_module.py geosuite/io geosmith/workflows/io workflows io")
        sys.exit(1)
    
    source_dir = Path(sys.argv[1])
    target_dir = Path(sys.argv[2])
    layer = sys.argv[3]
    domain = sys.argv[4] if len(sys.argv) > 4 else source_dir.name
    
    max_size = MAX_FILE_SIZES.get(domain, 600)
    
    print(f"Migrating {source_dir} → {target_dir}")
    print(f"Layer: {layer}, Domain: {domain}, Max size: {max_size} lines")
    
    result = batch_migrate_module(source_dir, target_dir, layer, domain, max_size)
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        sys.exit(1)
    
    print(f"\n✅ Migrated {result['migrated']} functions/classes")
    print(f"   Target: {result['target']}")
    
    if result.get("split_into"):
        print(f"   Split into: {', '.join(result['split_into'])}")
    
    if result.get("violations"):
        print(f"\n⚠️  {len(result['violations'])} architecture violations found:")
        for violation in result["violations"][:5]:
            print(f"   - {violation}")
        if len(result["violations"]) > 5:
            print(f"   ... and {len(result['violations']) - 5} more")


if __name__ == "__main__":
    main()

