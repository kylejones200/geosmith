#!/usr/bin/env python3
"""Batch migrate GeoSuite ML module to GeoSmith with 4-layer architecture.

Separates:
- Primitives: Pure operations (interpretability calculations, cross-validation, model utils)
- Workflows: Plotting functions (matplotlib-based)
- Tasks: User-facing models (check if already exist)
"""

import ast
from collections import defaultdict
from pathlib import Path

GEOSUITE_ML = Path("/Users/kylejonespatricia/geosuite/geosuite/ml")
GEOSMITH_PRIMITIVES = Path("geosmith/primitives")
GEOSMITH_WORKFLOWS = Path("geosmith/workflows")

# Layer 2: Primitives (pure operations, no plotting, no I/O)
PRIMITIVES_GROUPS = {
    "interpretability": {
        "patterns": ["get_feature_importance", "calculate_shap_values"],
        "description": "Model interpretability calculations (no plotting)",
        "forbidden": ["plot", "plt", "matplotlib", "figure"],
    },
    "cross_validation": {
        "patterns": ["WellBasedKFold", "SpatialCrossValidator", "cross", "cv"],
        "description": "Cross-validation schemes for geoscience data",
    },
    "model_utils": {
        "patterns": ["confusion_matrix", "compute_metrics", "display_cm", "_adjust"],
        "description": "Model evaluation utilities (calculations only, no plotting)",
        "forbidden": ["plot", "plt", "matplotlib", "signalplot"],
    },
    "hyperparameter": {
        "patterns": ["hyperparameter", "optimize", "SubsurfaceHyperparameter"],
        "description": "Hyperparameter optimization utilities",
    },
}

# Layer 4: Workflows (plotting functions)
WORKFLOWS_GROUPS = {
    "interpretability_plots": {
        "patterns": ["plot_feature_importance", "plot_shap_summary", "partial_dependence_plot"],
        "description": "Model interpretability plotting",
    },
    "confusion_matrix_plots": {
        "patterns": ["plot_confusion_matrix"],
        "description": "Confusion matrix visualization",
    },
}

# Layer 3: Tasks (check if these exist, may not need migration)
TASKS_TO_CHECK = ["FaciesTask", "ClusteringTask", "PermeabilityPredictor", "PorosityPredictor"]


def extract_functions_from_file(source_file: Path) -> list[dict]:
    """Extract all functions and classes from a Python file using AST."""
    try:
        content = source_file.read_text()
        tree = ast.parse(content)
        source_lines = content.splitlines()

        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                start_line = node.lineno - 1
                if node.decorator_list:
                    start_line = node.decorator_list[0].lineno - 1
                end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 50

                func_code = "\n".join(source_lines[start_line:end_line])

                # Check if it contains plotting (violates Layer 2)
                has_plotting = any(keyword in func_code.lower() for keyword in ["plt.", "matplotlib", "figure", "plot("])

                functions.append({
                    "name": node.name,
                    "type": "function",
                    "code": func_code,
                    "start_line": start_line,
                    "end_line": end_line,
                    "has_plotting": has_plotting,
                })
            elif isinstance(node, ast.ClassDef):
                start_line = node.lineno - 1
                if node.decorator_list:
                    start_line = node.decorator_list[0].lineno - 1
                end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 100

                class_code = "\n".join(source_lines[start_line:end_line])
                has_plotting = any(keyword in class_code.lower() for keyword in ["plt.", "matplotlib", "figure", "plot("])

                functions.append({
                    "name": node.name,
                    "type": "class",
                    "code": class_code,
                    "start_line": start_line,
                    "end_line": end_line,
                    "has_plotting": has_plotting,
                })

        return functions
    except Exception as e:
        print(f"⚠ Error parsing {source_file}: {e}")
        return []


def assign_to_layer(func_name: str, func_code: str, has_plotting: bool) -> tuple[str, str]:
    """Assign function to appropriate layer and module."""
    func_lower = func_name.lower()
    code_lower = func_code.lower()

    # Plotting functions → Workflows
    if has_plotting or any(keyword in code_lower for keyword in ["plt.", "matplotlib", "figure", "signalplot"]):
        for group, config in WORKFLOWS_GROUPS.items():
            if any(pattern.lower() in func_lower for pattern in config["patterns"]):
                return "workflows", group
        # Default to interpretability_plots if it's a plotting function
        if "plot" in func_lower or "shap" in func_lower or "partial" in func_lower:
            return "workflows", "interpretability_plots"
        return "workflows", "confusion_matrix_plots"

    # Pure operations → Primitives
    for group, config in PRIMITIVES_GROUPS.items():
        patterns = config.get("patterns", [])
        forbidden = config.get("forbidden", [])

        if any(keyword in code_lower for keyword in forbidden):
            continue

        if any(pattern.lower() in func_lower for pattern in patterns):
            return "primitives", group

    # Default: try to infer from name
    if "interpret" in func_lower or "importance" in func_lower or "shap" in func_lower:
        return "primitives", "interpretability"
    elif "cv" in func_lower or "cross" in func_lower or "fold" in func_lower:
        return "primitives", "cross_validation"
    elif "confusion" in func_lower or "metric" in func_lower:
        return "primitives", "model_utils"
    elif "optimize" in func_lower or "hyperparameter" in func_lower:
        return "primitives", "hyperparameter"

    return "primitives", "interpretability"  # Default fallback


def clean_code_for_layer(code: str, target_layer: str) -> str:
    """Clean code to comply with layer rules."""
    if target_layer == "primitives":
        # Remove plotting imports and calls
        lines = code.split("\n")
        cleaned = []
        skip_next = False

        for i, line in enumerate(lines):
            # Skip matplotlib imports
            if any(pattern in line for pattern in ["import matplotlib", "import plt", "import signalplot", "from matplotlib"]):
                continue
            # Skip signalplot.apply()
            if "signalplot.apply()" in line:
                continue
            # Skip plotting function definitions (but keep the logic)
            if line.strip().startswith("def plot_") and target_layer == "primitives":
                # Don't include plotting functions in primitives
                break

            # Remove plt. calls but keep return fig
            if "plt." in line and target_layer == "primitives":
                continue

            cleaned.append(line)

        return "\n".join(cleaned)

    return code


def generate_module_content(functions: list[dict], module_name: str, description: str, layer: str) -> str:
    """Generate content for a module."""
    header = f'''"""Geosmith ML: {description}

Migrated from geosuite.ml.
Layer {2 if layer == "primitives" else 4}: {'Primitives - Pure operations' if layer == 'primitives' else 'Workflows - Plotting and I/O'}.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

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

'''

    header += "import logging\n\n"
    header += "logger = logging.getLogger(__name__)\n\n"

    # Combine function sources (sorted by line number)
    funcs_sorted = sorted(functions, key=lambda x: x["start_line"])
    module_content = header + "\n\n\n".join(
        clean_code_for_layer(func["code"], layer) for func in funcs_sorted
    ) + "\n"

    return module_content


def main():
    """Main execution."""
    if not GEOSUITE_ML.exists():
        print(f"❌ GeoSuite ML directory not found: {GEOSUITE_ML}")
        return

    # Get all Python files from geosuite/ml
    source_files = [f for f in GEOSUITE_ML.glob("*.py") if f.name != "__init__.py"]

    all_functions = defaultdict(lambda: defaultdict(list))

    # Extract all functions
    for source_file in source_files:
        print(f"📄 Scanning {source_file.name}...")
        functions = extract_functions_from_file(source_file)

        for func in functions:
            layer, group = assign_to_layer(func["name"], func["code"], func["has_plotting"])
            all_functions[layer][group].append(func)

    # Create package directories
    ml_primitives_dir = GEOSMITH_PRIMITIVES / "ml"
    ml_primitives_dir.mkdir(exist_ok=True)

    ml_workflows_dir = GEOSMITH_WORKFLOWS / "plotting"
    ml_workflows_dir.mkdir(parents=True, exist_ok=True)

    # Generate primitives modules
    print("\n📦 Creating primitives modules...")
    for group, funcs in all_functions.get("primitives", {}).items():
        if not funcs:
            continue

        config = PRIMITIVES_GROUPS.get(group, {"description": group.replace("_", " ").title()})
        module_file = ml_primitives_dir / f"{group}.py"
        module_content = generate_module_content(funcs, group, config["description"], "primitives")
        module_file.write_text(module_content)
        print(f"   ✅ {module_file}: {len(funcs)} functions/classes")

    # Generate workflows modules (append to existing plotting.py if needed)
    print("\n📊 Creating workflows modules...")
    for group, funcs in all_functions.get("workflows", {}).items():
        if not funcs:
            continue

        config = WORKFLOWS_GROUPS.get(group, {"description": group.replace("_", " ").title()})
        module_file = ml_workflows_dir / f"ml_{group}.py"
        module_content = generate_module_content(funcs, group, config["description"], "workflows")
        module_file.write_text(module_content)
        print(f"   ✅ {module_file}: {len(funcs)} functions/classes")

    # Generate __init__.py for primitives/ml
    init_content = generate_primitives_init(all_functions.get("primitives", {}))
    (ml_primitives_dir / "__init__.py").write_text(init_content)
    print(f"\n   ✅ {ml_primitives_dir / '__init__.py'}")

    print("\n✅ ML module migration complete!")


def generate_primitives_init(module_assignments: dict) -> str:
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

    # Import from submodules
    for group, funcs in module_assignments.items():
        if not funcs:
            continue

        func_names = [f['name'] for f in funcs]
        if func_names:
            content += f"# {group.replace('_', ' ').title()}\n"
            content += f"from geosmith.primitives.ml.{group} import (\n"
            for name in sorted(func_names):
                content += f"    {name},\n"
            content += ")\n\n"

    # Generate __all__
    all_names = [f['name'] for funcs in module_assignments.values() for f in funcs]
    content += f"\n__all__ = {sorted(all_names)!r}\n"

    return content


if __name__ == "__main__":
    main()


