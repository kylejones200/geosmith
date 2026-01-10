#!/usr/bin/env python3
"""Smart Migration Planner - Analyzes geosuite and creates migration plan.

Maintains 4-layer architecture and modularity constraints.
"""

import ast
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

GEOSUITE_ROOT = Path("/Users/kylejonespatricia/geosuite")
GEOSMITH_ROOT = Path(__file__).parent.parent

# Layer mapping: geosuite module -> geosmith layer
LAYER_MAPPING = {
    "objects": "geosmith.objects",
    "primitives": "geosmith.primitives",
    "tasks": "geosmith.tasks",
    "workflows": "geosmith.workflows",
}

# Domain -> target module mapping (maintains modularity)
DOMAIN_MODULES = {
    "petro": {
        "target": "geosmith/primitives/petrophysics",
        "max_file_size": 600,  # lines
        "split_into": ["water_saturation", "permeability", "rock_physics", "avp", "lithology"],
    },
    "geomech": {
        "target": "geosmith/primitives/geomechanics",
        "max_file_size": 600,
        "split_into": ["stress", "pressure", "failure", "fracture", "inversion", "wellbore", "field", "parallel"],
        "status": "migrated",  # Already done!
    },
    "geospatial": {
        "target": "geosmith/primitives/geometry",
        "max_file_size": 500,
        "split_into": ["distance", "neighbors", "polygons", "intersection"],
    },
    "io": {
        "target": "geosmith/workflows",
        "max_file_size": 700,
        "split_into": ["vector_io", "raster_io", "witsml", "dlis", "resqml", "ppdm", "csv"],
    },
    "forecasting": {
        "target": "geosmith/tasks",
        "max_file_size": 500,
        "split_into": ["decline", "production"],
        "status": "migrated",  # Decline curve analysis done
    },
    "mining": {
        "target": "geosmith/primitives/mining",
        "max_file_size": 600,
        "split_into": ["block_model", "variogram", "kriging", "simulation"],
    },
    "plotting": {
        "target": "geosmith/workflows/plotting",
        "max_file_size": 700,
        "split_into": ["well_logs", "maps", "crossplots", "interactive"],
    },
    "stratigraphy": {
        "target": "geosmith/tasks/stratigraphy",
        "max_file_size": 400,
        "status": "migrated",
    },
    "ml": {
        "target": "geosmith/primitives/ml",
        "max_file_size": 500,
        "split_into": ["interpretability", "uncertainty", "features"],
    },
    "utils": {
        "target": "geosmith/primitives/utils",
        "max_file_size": 400,
        "split_into": ["validation", "transforms"],
    },
}


def scan_module(module_path: Path) -> Dict[str, List[Tuple[str, int]]]:
    """Scan a Python module and extract all functions/classes.
    
    Returns: Dict mapping file_name -> list of (name, line_count) tuples
    """
    functions = defaultdict(list)
    
    if not module_path.exists():
        return functions
    
    for py_file in module_path.rglob("*.py"):
        if "__pycache__" in str(py_file) or "test" in str(py_file).lower():
            continue
        
        try:
            content = py_file.read_text()
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    line_count = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 50
                    functions[py_file.name].append((node.name, line_count))
                elif isinstance(node, ast.ClassDef):
                    line_count = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 100
                    functions[py_file.name].append((node.name, line_count))
        except SyntaxError:
            continue
    
    return functions


def analyze_geosuite() -> Dict[str, Dict]:
    """Analyze geosuite structure and create migration plan."""
    geosuite_dir = GEOSUITE_ROOT / "geosuite"
    
    if not geosuite_dir.exists():
        print(f"⚠ GeoSuite directory not found: {geosuite_dir}")
        return {}
    
    analysis = {}
    
    for domain, config in DOMAIN_MODULES.items():
        if config.get("status") == "migrated":
            continue
        
        domain_path = geosuite_dir / domain
        if not domain_path.exists():
            continue
        
        functions = scan_module(domain_path)
        total_functions = sum(len(funcs) for funcs in functions.values())
        total_files = len(functions)
        
        # Calculate estimated lines
        total_lines = sum(sum(lines for _, lines in funcs) for funcs in functions.values())
        
        # Check if target file exists and its size
        target_path = GEOSMITH_ROOT / config["target"]
        target_size = 0
        if target_path.exists() and target_path.is_file():
            target_size = len(target_path.read_text().splitlines())
        
        analysis[domain] = {
            "source_path": str(domain_path),
            "target_path": config["target"],
            "total_files": total_files,
            "total_functions": total_functions,
            "estimated_lines": total_lines,
            "target_current_size": target_size,
            "max_file_size": config["max_file_size"],
            "needs_splitting": target_size > config["max_file_size"],
            "split_modules": config.get("split_into", []),
            "functions": dict(functions),
        }
    
    return analysis


def generate_migration_plan(analysis: Dict[str, Dict]) -> str:
    """Generate migration plan markdown."""
    plan = """# Smart Migration Plan

## Strategy: Fast, Modular, Architecture-Preserving

### Principles:
1. **Batch by domain** - Migrate entire modules at once
2. **Maintain modularity** - Keep files < 600 lines
3. **Preserve 4-layer architecture** - Objects → Primitives → Tasks → Workflows
4. **Defer tests** - Migrate code first, test in batches
5. **Use automation** - Leverage refactoring scripts

## Priority Order (Largest Impact, Fastest Wins)

"""
    
    # Sort by priority: largest modules that need migration
    priority_order = sorted(
        analysis.items(),
        key=lambda x: (x[1]["total_functions"], -x[1].get("target_current_size", 0)),
        reverse=True
    )
    
    for i, (domain, info) in enumerate(priority_order, 1):
        needs_splitting = "⚠️ **NEEDS REFACTORING**" if info["needs_splitting"] else "✅ OK"
        plan += f"""
### {i}. {domain.upper()} Module
- **Source**: `{info['source_path']}`
- **Target**: `{info['target_path']}`
- **Functions**: {info['total_functions']} functions/classes in {info['total_files']} files
- **Estimated lines**: ~{info['estimated_lines']} lines
- **Current target size**: {info['target_current_size']} lines {needs_splitting}
- **Max allowed**: {info['max_file_size']} lines
- **Strategy**: {'Split into package structure' if info['needs_splitting'] else 'Batch migrate to single module'}
"""
        
        if info["needs_splitting"]:
            plan += f"  - Split target into: {', '.join(info['split_modules'])}\n"
        
        plan += f"""
#### Migration Steps:
1. Scan source files: `grep -r "^def\\|^class" {info['source_path']}`
2. Migrate all functions to target module(s)
3. Update `__init__.py` exports
4. Single commit: "Migrate {domain} module from GeoSuite"
5. Write batch tests (defer to end)
"""
    
    plan += """
## Batch Migration Script Template

Use `scripts/refactor_geomechanics_ast.py` as template for automated extraction.

## File Size Checks

Before migrating, check current file sizes:
```bash
find geosmith/primitives -name "*.py" -exec wc -l {} \; | awk '$1 > 500 {print $1, $2}'
```

## Next Immediate Actions

1. **Fix petrophysics.py** (1432 lines → split into package)
2. **Fix production.py** (671 lines → split if needed)
3. **Migrate geosuite.io** (highest priority, many utilities)
4. **Migrate geosuite.geospatial** (core geometry operations)
5. **Migrate geosuite.mining** (block models, variograms already done, add remaining)

"""
    
    return plan


def main():
    """Main execution."""
    print("Analyzing GeoSuite structure...")
    analysis = analyze_geosuite()
    
    print(f"\nFound {len(analysis)} domains to analyze")
    
    plan = generate_migration_plan(analysis)
    
    output_file = GEOSMITH_ROOT / "SMART_MIGRATION_PLAN.md"
    output_file.write_text(plan)
    
    print(f"\n✅ Migration plan written to: {output_file}")
    print(f"\nSummary:")
    for domain, info in sorted(analysis.items(), key=lambda x: x[1]["total_functions"], reverse=True):
        status = "⚠️ NEEDS REFACTORING" if info["needs_splitting"] else "✅ Ready"
        print(f"  {domain:20s}: {info['total_functions']:3d} functions, {status}")


if __name__ == "__main__":
    main()

