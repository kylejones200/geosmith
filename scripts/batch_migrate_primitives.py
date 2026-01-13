#!/usr/bin/env python3
"""Helper script to batch migrate primitive functions from GeoSuite.

This script helps identify and migrate multiple functions at once.
Usage: python scripts/batch_migrate_primitives.py geosuite.petro.permeability
"""

import ast
import sys
from pathlib import Path


def extract_function_signatures(file_path: Path) -> list[dict]:
    """Extract function signatures and docstrings from a Python file."""
    with open(file_path) as f:
        tree = ast.parse(f.read())

    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Extract function signature
            args = [arg.arg for arg in node.args.args]

            # Extract docstring
            docstring = ast.get_docstring(node) or ""

            # Check for optional dependencies
            body_code = ast.unparse(node.body) if hasattr(ast, "unparse") else ""
            has_scipy = "scipy" in body_code.lower()
            has_sklearn = "sklearn" in body_code.lower() or "sklearn" in body_code.lower()
            has_numba = "@njit" in ast.unparse(node.decorator_list) if hasattr(ast, "unparse") else False

            functions.append({
                "name": node.name,
                "args": args,
                "docstring": docstring[:200] if docstring else "",  # First 200 chars
                "has_scipy": has_scipy,
                "has_sklearn": has_sklearn,
                "has_numba": has_numba,
                "is_private": node.name.startswith("_"),
            })

    return functions


def generate_migration_template(functions: list[dict], module_name: str) -> str:
    """Generate a template for migrating functions."""
    public_functions = [f for f in functions if not f["is_private"]]

    template = f"""\"\"\"Migrated from {module_name}\"\"\"

import numpy as np
from typing import Union, Optional

# Optional dependencies
try:
    from scipy import ...
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn import ...
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])

"""

    for func in public_functions:
        template += f"""
def {func["name"]}(
    # TODO: Add parameters
) -> Union[np.ndarray, float]:
    \"\"\"{func["docstring"]}

    Migrated from {module_name}.{func["name"]}
    \"\"\"
    # TODO: Implement
    pass
"""

    return template


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python batch_migrate_primitives.py <source_file>")
        sys.exit(1)

    source_file = Path(sys.argv[1])
    if not source_file.exists():
        # Try relative to geosuite
        source_file = Path("/Users/kylejonespatricia/geosuite") / sys.argv[1].replace(".", "/") / "__init__.py"
        if not source_file.exists():
            source_file = Path("/Users/kylejonespatricia/geosuite") / sys.argv[1].replace(".", "/") + ".py"

    if not source_file.exists():
        print(f"File not found: {source_file}")
        sys.exit(1)

    functions = extract_function_signatures(source_file)
    print(f"Found {len(functions)} functions in {source_file}")
    print(f"  Public: {len([f for f in functions if not f['is_private']])}")
    print(f"  Private: {len([f for f in functions if f['is_private']])}")

    if "--template" in sys.argv:
        print("\n" + generate_migration_template(functions, sys.argv[1]))


