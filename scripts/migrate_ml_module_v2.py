#!/usr/bin/env python3
"""Batch migrate GeoSuite ML module to GeoSmith with proper 4-layer architecture.

Strategy:
1. Extract functions from specific source files (not by name pattern)
2. Separate plotting functions (workflows) from pure operations (primitives)
3. Keep interpretability.py small (< 500 lines) - only pure calculations
4. Put plotting functions in workflows/plotting/
"""

import ast
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

GEOSUITE_ML = Path("/Users/kylejonespatricia/geosuite/geosuite/ml")
GEOSMITH_PRIMITIVES = Path("geosmith/primitives")
GEOSMITH_WORKFLOWS = Path("geosmith/workflows")

# Map source files to target modules
FILE_MAPPING = {
    "interpretability.py": {
        "primitives": "interpretability",  # Pure functions: get_feature_importance, calculate_shap_values
        "workflows": "ml_interpretability_plots",  # Plotting: plot_feature_importance, plot_shap_summary, partial_dependence_plot
    },
    "cross_validation.py": {
        "primitives": "cross_validation",  # All functions
    },
    "confusion_matrix_utils.py": {
        "primitives": "model_utils",  # Pure functions: display_cm, display_adj_cm, compute_metrics_from_cm, confusion_matrix_to_dataframe, _adjust_confusion_matrix_kernel
        "workflows": "ml_confusion_matrix_plots",  # Plotting: plot_confusion_matrix
    },
    "hyperparameter_optimization.py": {
        "primitives": "hyperparameter",  # All functions (but check for plotting)
    },
    # These might belong in tasks layer, skip for now or mark for review
    "classifiers.py": "skip",  # train_and_predict, FaciesResult - might be in tasks
    "regression.py": "skip",  # PermeabilityPredictor, PorosityPredictor - might be in tasks
    "clustering.py": "skip",  # FaciesClusterer - already in tasks?
    "deep_models.py": "skip",  # DeepFaciesClassifier, DeepPropertyPredictor - might be in tasks
    "enhanced_classifiers.py": "skip",  # MLflowFaciesClassifier - might be in tasks
}


def extract_top_level_functions(source_file: Path) -> List[Dict]:
    """Extract only top-level functions and classes from a Python file."""
    try:
        content = source_file.read_text()
        tree = ast.parse(content)
        source_lines = content.splitlines()
        
        functions = []
        for node in ast.walk(tree):
            # Only extract top-level functions/classes (module-level)
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
                
                # Check if it contains plotting
                code_lower = func_code.lower()
                has_plotting = any(
                    keyword in code_lower
                    for keyword in ["plt.", "matplotlib", "figure(", "signalplot", "plot("]
                ) and "def plot_" in func_code.lower()
                
                functions.append({
                    "name": node.name,
                    "type": "class" if isinstance(node, ast.ClassDef) else "function",
                    "code": func_code,
                    "start_line": start_line,
                    "end_line": end_line,
                    "has_plotting": has_plotting,
                    "source_file": source_file.name,
                })
        
        return functions
    except Exception as e:
        print(f"⚠ Error parsing {source_file}: {e}")
        return []


def separate_plotting_functions(functions: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Separate plotting functions from pure operations."""
    pure = []
    plotting = []
    
    for func in functions:
        if func["has_plotting"] or func["name"].startswith("plot_"):
            plotting.append(func)
        else:
            pure.append(func)
    
    return pure, plotting


def generate_module_header(module_name: str, description: str, layer: str) -> str:
    """Generate module header."""
    layer_num = 2 if layer == "primitives" else 4
    layer_desc = "Primitives - Pure operations" if layer == "primitives" else "Workflows - Plotting and I/O"
    
    header = f'''"""Geosmith ML: {description}

Migrated from geosuite.ml.
Layer {layer_num}: {layer_desc}.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import logging

logger = logging.getLogger(__name__)

'''
    
    if layer == "primitives":
        header += '''
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None  # type: ignore

try:
    from sklearn.model_selection import BaseCrossValidator
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    BaseCrossValidator = None  # type: ignore

try:
    from shap import TreeExplainer, KernelExplainer
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    TreeExplainer = None  # type: ignore
    KernelExplainer = None  # type: ignore

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
    elif layer == "workflows":
        header += '''
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None  # type: ignore

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None  # type: ignore
    Figure = None  # type: ignore

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None  # type: ignore

try:
    import signalplot
    SIGNALPLOT_AVAILABLE = True
except ImportError:
    SIGNALPLOT_AVAILABLE = False
    signalplot = None  # type: ignore

'''
    
    return header


def main():
    """Main execution."""
    if not GEOSUITE_ML.exists():
        print(f"❌ GeoSuite ML directory not found: {GEOSUITE_ML}")
        return
    
    # Create directories
    ml_primitives_dir = GEOSMITH_PRIMITIVES / "ml"
    ml_primitives_dir.mkdir(exist_ok=True)
    
    ml_workflows_dir = GEOSMITH_WORKFLOWS / "plotting"
    ml_workflows_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each source file
    primitives_modules = defaultdict(list)
    workflows_modules = defaultdict(list)
    
    for source_file in sorted(GEOSUITE_ML.glob("*.py")):
        if source_file.name == "__init__.py":
            continue
        
        mapping = FILE_MAPPING.get(source_file.name, None)
        if mapping == "skip":
            print(f"⏭ Skipping {source_file.name} (may belong in tasks layer)")
            continue
        
        if mapping is None:
            print(f"⚠ No mapping for {source_file.name}, skipping")
            continue
        
        print(f"📄 Processing {source_file.name}...")
        functions = extract_top_level_functions(source_file)
        
        if "primitives" in mapping:
            pure, plotting = separate_plotting_functions(functions)
            primitives_modules[mapping["primitives"]].extend(pure)
            if plotting:
                if "workflows" in mapping:
                    workflows_modules[mapping["workflows"]].extend(plotting)
        
        elif "workflows" in mapping:
            workflows_modules[mapping["workflows"]].extend(functions)
    
    # Generate primitives modules
    print("\n📦 Creating primitives modules...")
    for module_name, funcs in primitives_modules.items():
        if not funcs:
            continue
        
        funcs_sorted = sorted(funcs, key=lambda x: x["start_line"])
        descriptions = {
            "interpretability": "Model interpretability calculations (feature importance, SHAP)",
            "cross_validation": "Cross-validation schemes for geoscience data",
            "model_utils": "Model evaluation utilities (confusion matrix calculations)",
            "hyperparameter": "Hyperparameter optimization utilities",
        }
        
        header = generate_module_header(module_name, descriptions.get(module_name, module_name), "primitives")
        module_content = header + "\n\n\n".join(func["code"] for func in funcs_sorted) + "\n"
        
        module_file = ml_primitives_dir / f"{module_name}.py"
        module_file.write_text(module_content)
        print(f"   ✅ {module_file}: {len(funcs)} functions/classes ({module_file.stat().st_size // 1024}KB)")
    
    # Generate workflows modules
    print("\n📊 Creating workflows modules...")
    for module_name, funcs in workflows_modules.items():
        if not funcs:
            continue
        
        funcs_sorted = sorted(funcs, key=lambda x: x["start_line"])
        descriptions = {
            "ml_interpretability_plots": "Model interpretability plotting",
            "ml_confusion_matrix_plots": "Confusion matrix visualization",
        }
        
        header = generate_module_header(module_name, descriptions.get(module_name, module_name), "workflows")
        module_content = header + "\n\n\n".join(func["code"] for func in funcs_sorted) + "\n"
        
        module_file = ml_workflows_dir / f"{module_name}.py"
        module_file.write_text(module_content)
        print(f"   ✅ {module_file}: {len(funcs)} functions/classes ({module_file.stat().st_size // 1024}KB)")
    
    # Generate __init__.py for primitives/ml
    init_content = generate_primitives_init(primitives_modules)
    (ml_primitives_dir / "__init__.py").write_text(init_content)
    print(f"\n   ✅ {ml_primitives_dir / '__init__.py'}")
    
    print("\n✅ ML module migration complete!")


def generate_primitives_init(module_assignments: Dict) -> str:
    """Generate __init__.py for primitives/ml package."""
    content = '''"""Geosmith ML primitives (modular package).

Pure ML operations split into logical modules:
- interpretability: Model interpretability calculations (feature importance, SHAP)
- cross_validation: Cross-validation schemes for geoscience data
- model_utils: Model evaluation utilities (confusion matrix calculations)
- hyperparameter: Hyperparameter optimization utilities

This package maintains backward compatibility with the original flat import:
`from geosmith.primitives.ml import ...`
"""

# Import order matters - avoid circular imports

'''
    
    for module_name, funcs in sorted(module_assignments.items()):
        if not funcs:
            continue
        
        func_names = sorted(set(f['name'] for f in funcs))
        content += f"# {module_name.replace('_', ' ').title()}\n"
        content += f"from geosmith.primitives.ml.{module_name} import (\n"
        for name in func_names:
            # Skip special methods and private helpers
            if name.startswith("__") and name != "__init__":
                continue
            if name.startswith("_") and name != "_adjust_confusion_matrix_kernel":
                continue
            content += f"    {name},\n"
        content += ")\n\n"
    
    # Generate __all__
    all_names = []
    for funcs in module_assignments.values():
        for func in funcs:
            name = func['name']
            if not name.startswith("__") or name == "__init__":
                if not name.startswith("_") or name == "_adjust_confusion_matrix_kernel":
                    all_names.append(name)
    
    content += f"__all__ = {sorted(set(all_names))!r}\n"
    
    return content


if __name__ == "__main__":
    main()

