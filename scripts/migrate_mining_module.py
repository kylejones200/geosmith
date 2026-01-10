#!/usr/bin/env python3
"""Batch migrate remaining GeoSuite mining module to GeoSmith.

Focuses on:
- ore_modeling.py (Hybrid IDW+ML model)
- forecasting.py (Ore grade forecasting methods)

Note: block_model.py, drillhole.py, features.py, geostatistics.py, interpolation.py already migrated.
"""

import ast
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

GEOSUITE_MINING = Path("/Users/kylejonespatricia/geosuite/geosuite/mining")
GEOSMITH_PRIMITIVES = Path("geosmith/primitives")

# Map source files to target modules
FILE_MAPPING = {
    "ore_modeling.py": {
        "target": "ore_modeling",
        "description": "Hybrid IDW+ML ore grade estimation",
    },
    "forecasting.py": {
        "target": "forecasting",
        "description": "Ore grade forecasting (Kriging, GPR, XGBoost)",
    },
    # Already migrated - skip
    "block_model.py": "skip",  # In tasks/blockmodeltask.py
    "drillhole.py": "skip",  # In workflows/drillhole.py
    "features.py": "skip",  # In primitives/features.py
    "geostatistics.py": "skip",  # In primitives/variogram.py, kriging.py, simulation.py
    "interpolation.py": "skip",  # In primitives/interpolation.py
}


def extract_top_level_functions(source_file: Path) -> List[Dict]:
    """Extract only top-level functions and classes from a Python file."""
    try:
        content = source_file.read_text()
        tree = ast.parse(content)
        source_lines = content.splitlines()
        
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                # Check if it's a top-level definition (not nested)
                parent = getattr(node, 'parent', None)
                if parent and isinstance(parent, (ast.FunctionDef, ast.ClassDef)):
                    continue  # Skip nested definitions
                
                # Assign parent for nested detection
                for child in ast.iter_child_nodes(node):
                    child.parent = node
                
                start_line = node.lineno - 1
                if node.decorator_list:
                    start_line = node.decorator_list[0].lineno - 1
                end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 50
                
                func_code = "\n".join(source_lines[start_line:end_line])
                
                # Check for plotting/geopandas (violates Layer 2)
                code_lower = func_code.lower()
                has_plotting = any(
                    keyword in code_lower
                    for keyword in ["plt.", "matplotlib", "figure(", "signalplot", "plot("]
                ) and "def plot_" in func_code.lower()
                has_geopandas = "geopandas" in code_lower or "gpd." in code_lower or "gdf" in code_lower.lower()
                
                functions.append({
                    "name": node.name,
                    "type": "class" if isinstance(node, ast.ClassDef) else "function",
                    "code": func_code,
                    "start_line": start_line,
                    "end_line": end_line,
                    "has_plotting": has_plotting,
                    "has_geopandas": has_geopandas,
                })
        
        return functions
    except Exception as e:
        print(f"⚠ Error parsing {source_file}: {e}")
        return []


def clean_imports_for_primitives(code: str, source_file: str) -> str:
    """Clean imports for primitives layer (Layer 2)."""
    lines = code.splitlines()
    cleaned = []
    skip_imports = False
    
    for i, line in enumerate(lines):
        # Skip plotting imports
        if any(pattern in line for pattern in ["import matplotlib", "import plt", "import signalplot", "from matplotlib"]):
            continue
        
        # Keep geopandas imports but wrap in try/except (for forecasting.py)
        if "import geopandas" in line or "from geopandas" in line:
            # Replace with optional import pattern
            cleaned.append("try:")
            cleaned.append("    import geopandas as gpd")
            cleaned.append("    GEOPANDAS_AVAILABLE = True")
            cleaned.append("except ImportError:")
            cleaned.append("    GEOPANDAS_AVAILABLE = False")
            cleaned.append("    gpd = None  # type: ignore")
            continue
        
        # Update relative imports to absolute
        if "from .interpolation" in line:
            cleaned.append("from geosmith.primitives.interpolation import idw_interpolate, compute_idw_residuals")
            continue
        if "from .features" in line:
            cleaned.append("from geosmith.primitives.features import build_spatial_features, build_block_model_features")
            continue
        if "from .geostatistics" in line:
            # These are already migrated to variogram, kriging, simulation
            cleaned.append("# Variogram/kriging already migrated to primitives.variogram, primitives.kriging, primitives.simulation")
            continue
        
        cleaned.append(line)
    
    return "\n".join(cleaned)


def generate_module_header(module_name: str, description: str) -> str:
    """Generate module header for primitives layer."""
    return f'''"""Geosmith mining: {description}

Migrated from geosuite.mining.
Layer 2: Primitives - Pure operations.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None  # type: ignore

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    GradientBoostingRegressor = None  # type: ignore
    RandomForestRegressor = None  # type: ignore
    GroupKFold = None  # type: ignore
    mean_absolute_error = None  # type: ignore
    r2_score = None  # type: ignore
    mean_squared_error = None  # type: ignore

try:
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
    SKLEARN_PREPROCESSING_AVAILABLE = True
except ImportError:
    SKLEARN_PREPROCESSING_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None  # type: ignore

try:
    from skgstat import Variogram
    SKGSTAT_AVAILABLE = True
except ImportError:
    SKGSTAT_AVAILABLE = False
    Variogram = None  # type: ignore

try:
    from pykrige.ok import OrdinaryKriging
    PYKRIGE_AVAILABLE = True
except ImportError:
    PYKRIGE_AVAILABLE = False
    OrdinaryKriging = None  # type: ignore

try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    cKDTree = None  # type: ignore

try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    gpd = None  # type: ignore

'''


def main():
    """Main execution."""
    if not GEOSUITE_MINING.exists():
        print(f"❌ GeoSuite mining directory not found: {GEOSUITE_MINING}")
        return
    
    # Create directory
    mining_dir = GEOSMITH_PRIMITIVES / "mining"
    mining_dir.mkdir(exist_ok=True)
    
    modules_to_migrate = {}
    
    # Process each source file
    for source_file in sorted(GEOSUITE_MINING.glob("*.py")):
        if source_file.name == "__init__.py":
            continue
        
        mapping = FILE_MAPPING.get(source_file.name, None)
        if mapping == "skip":
            print(f"⏭ Skipping {source_file.name} (already migrated)")
            continue
        
        if mapping is None:
            print(f"⚠ No mapping for {source_file.name}, skipping")
            continue
        
        print(f"📄 Processing {source_file.name}...")
        functions = extract_top_level_functions(source_file)
        
        modules_to_migrate[mapping["target"]] = {
            "functions": functions,
            "description": mapping["description"],
        }
    
    # Generate modules
    print("\n📦 Creating primitives/mining modules...")
    for module_name, data in modules_to_migrate.items():
        funcs = data["functions"]
        if not funcs:
            continue
        
        funcs_sorted = sorted(funcs, key=lambda x: x["start_line"])
        
        header = generate_module_header(module_name, data["description"])
        
        # Clean code for each function
        cleaned_codes = []
        for func in funcs_sorted:
            cleaned_code = clean_imports_for_primitives(func["code"], f"geosuite/mining/{module_name}.py")
            cleaned_codes.append(cleaned_code)
        
        module_content = header + "\n\n\n".join(cleaned_codes) + "\n"
        
        module_file = mining_dir / f"{module_name}.py"
        module_file.write_text(module_content)
        print(f"   ✅ {module_file}: {len(funcs)} functions/classes ({len(module_content.splitlines())} lines)")
    
    # Generate __init__.py
    init_content = generate_mining_init(modules_to_migrate)
    (mining_dir / "__init__.py").write_text(init_content)
    print(f"\n   ✅ {mining_dir / '__init__.py'}")
    
    print("\n✅ Mining module migration complete!")


def generate_mining_init(module_assignments: Dict) -> str:
    """Generate __init__.py for primitives/mining package."""
    content = '''"""Geosmith mining primitives (modular package).

Ore modeling and forecasting operations split into logical modules:
- ore_modeling: Hybrid IDW+ML ore grade estimation
- forecasting: Ore grade forecasting (Kriging, GPR, XGBoost)

Note: Block model operations are in tasks/blockmodeltask.py
      Variogram/kriging/simulation are in primitives/variogram.py, kriging.py, simulation.py
      Feature engineering is in primitives/features.py

This package maintains backward compatibility with the original flat import:
`from geosmith.primitives.mining import ...`
"""

# Import order matters - avoid circular imports

'''
    
    for module_name, data in sorted(module_assignments.items()):
        funcs = data["functions"]
        if not funcs:
            continue
        
        func_names = sorted(set(f['name'] for f in funcs))
        content += f"# {module_name.replace('_', ' ').title()}\n"
        content += f"from geosmith.primitives.mining.{module_name} import (\n"
        for name in func_names:
            # Skip special methods and private helpers (unless they're exported)
            if name.startswith("__") and name != "__init__":
                continue
            if name.startswith("_") and not name.startswith("_"):
                # Only include public functions/classes
                continue
            content += f"    {name},\n"
        content += ")\n\n"
    
    # Generate __all__
    all_names = []
    for module_name, data in module_assignments.items():
        for func in data["functions"]:
            name = func['name']
            if not name.startswith("__") or name == "__init__":
                if not name.startswith("_"):  # Only public
                    all_names.append(name)
    
    content += f"__all__ = {sorted(set(all_names))!r}\n"
    
    return content


if __name__ == "__main__":
    main()

