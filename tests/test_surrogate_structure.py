"""Structural tests for surrogate models (no dependencies required).

Tests the code structure, imports, and basic functionality without requiring
scikit-learn or xgboost to be installed.
"""

import ast
import sys
from pathlib import Path


def test_surrogate_module_structure():
    """Test that surrogate.py has correct structure."""
    surrogate_path = Path(__file__).parent.parent / "geosmith" / "primitives" / "surrogate.py"

    assert surrogate_path.exists(), "surrogate.py file does not exist"

    # Parse AST to check structure
    with open(surrogate_path) as f:
        code = f.read()
        tree = ast.parse(code)

    # Check for required classes
    class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    assert "SurrogateModel" in class_names, "SurrogateModel class not found"
    assert "SurrogateMetrics" in class_names, "SurrogateMetrics class not found"

    # Check for required functions
    function_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    assert "train_simulation_emulator" in function_names, "train_simulation_emulator function not found"

    # Check for required methods in SurrogateModel
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "SurrogateModel":
            method_names = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
            assert "fit" in method_names, "fit method not found in SurrogateModel"
            assert "predict" in method_names, "predict method not found in SurrogateModel"
            assert "_create_model" in method_names, "_create_model method not found"
            assert "_prepare_training_data" in method_names, "_prepare_training_data method not found"

    print("✅ SurrogateModel structure is correct")


def test_surrogate_task_structure():
    """Test that surrogatetask.py has correct structure."""
    task_path = Path(__file__).parent.parent / "geosmith" / "tasks" / "surrogatetask.py"

    assert task_path.exists(), "surrogatetask.py file does not exist"

    # Parse AST
    with open(task_path) as f:
        code = f.read()
        tree = ast.parse(code)

    # Check for SurrogateTask class
    class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    assert "SurrogateTask" in class_names, "SurrogateTask class not found"

    # Check for required methods
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "SurrogateTask":
            method_names = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
            assert "train_emulator" in method_names, "train_emulator method not found"
            assert "predict" in method_names, "predict method not found"
            assert "validate_emulator" in method_names, "validate_emulator method not found"

    print("✅ SurrogateTask structure is correct")


def test_surrogate_imports():
    """Test that surrogate module can be imported (if dependencies available)."""
    try:
        from geosmith.primitives.surrogate import SurrogateModel, SurrogateMetrics, train_simulation_emulator

        print("✅ Surrogate module imports successfully (dependencies available)")
        return True
    except ImportError as e:
        print(f"⚠️  Surrogate module cannot be imported (dependencies missing): {e}")
        print("   This is expected if scikit-learn/xgboost are not installed")
        return False


def test_surrogate_task_imports():
    """Test that surrogate task can be imported (if dependencies available)."""
    try:
        from geosmith.tasks.surrogatetask import SurrogateTask

        print("✅ SurrogateTask imports successfully (dependencies available)")
        return True
    except ImportError as e:
        print(f"⚠️  SurrogateTask cannot be imported (dependencies missing): {e}")
        print("   This is expected if scikit-learn/xgboost are not installed")
        return False


def test_surrogate_code_syntax():
    """Test that surrogate code has valid Python syntax."""
    surrogate_path = Path(__file__).parent.parent / "geosmith" / "primitives" / "surrogate.py"
    task_path = Path(__file__).parent.parent / "geosmith" / "tasks" / "surrogatetask.py"

    for path in [surrogate_path, task_path]:
        with open(path) as f:
            code = f.read()
            try:
                ast.parse(code)
                print(f"✅ {path.name} has valid Python syntax")
            except SyntaxError as e:
                print(f"❌ {path.name} has syntax errors: {e}")
                raise


def test_surrogate_inheritance():
    """Test that SurrogateModel inherits from BaseSpatialModel."""
    surrogate_path = Path(__file__).parent.parent / "geosmith" / "primitives" / "surrogate.py"

    with open(surrogate_path) as f:
        lines = f.readlines()

    # Check for BaseSpatialModel import
    found_import = False
    found_inheritance = False
    
    for i, line in enumerate(lines):
        if "BaseSpatialModel" in line and "import" in line:
            found_import = True
        if "class SurrogateModel" in line and "BaseSpatialModel" in line:
            found_inheritance = True
    
    assert found_import, "BaseSpatialModel not imported"
    assert found_inheritance, "SurrogateModel does not inherit from BaseSpatialModel"

    print("✅ SurrogateModel inherits from BaseSpatialModel")


def main():
    """Run all structural tests."""
    print("=" * 70)
    print("Surrogate Model Structure Tests")
    print("=" * 70)
    print()

    tests = [
        test_surrogate_code_syntax,
        test_surrogate_module_structure,
        test_surrogate_task_structure,
        test_surrogate_inheritance,
        test_surrogate_imports,
        test_surrogate_task_imports,
    ]

    passed = 0
    for test in tests:
        try:
            result = test()
            if result is not False:
                passed += 1
        except AssertionError as e:
            print(f"❌ {test.__name__} failed: {e}")
        except Exception as e:
            print(f"❌ {test.__name__} raised exception: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 70)
    print(f"Tests passed: {passed}/{len(tests)}")
    print("=" * 70)


if __name__ == "__main__":
    main()

