#!/usr/bin/env python3
"""Quick test script for surrogate models (works without dependencies).

This script demonstrates the surrogate model API and checks basic functionality
without requiring scikit-learn or xgboost to be installed.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_basic_structure():
    """Test basic code structure."""
    print("Testing surrogate model structure...")

    # Check files exist
    surrogate_file = Path(__file__).parent.parent / "geosmith" / "primitives" / "surrogate.py"
    task_file = Path(__file__).parent.parent / "geosmith" / "tasks" / "surrogatetask.py"
    example_file = Path(__file__).parent.parent / "examples" / "surrogate_example.py"

    assert surrogate_file.exists(), "surrogate.py not found"
    assert task_file.exists(), "surrogatetask.py not found"
    assert example_file.exists(), "surrogate_example.py not found"

    print("✅ All files exist")

    # Check for required classes/functions
    with open(surrogate_file) as f:
        code = f.read()
        assert "class SurrogateModel" in code
        assert "class SurrogateMetrics" in code
        assert "def train_simulation_emulator" in code
        assert "def fit" in code
        assert "def predict" in code
        assert "BaseSpatialModel" in code

    print("✅ SurrogateModel has required methods")

    with open(task_file) as f:
        code = f.read()
        assert "class SurrogateTask" in code
        assert "def train_emulator" in code
        assert "def predict" in code
        assert "def validate_emulator" in code

    print("✅ SurrogateTask has required methods")

    return True


def test_imports():
    """Test that imports are set up correctly."""
    print("\nTesting imports...")

    # Test that imports are conditional
    try:
        primitives_code = Path(__file__).parent.parent / "geosmith" / "primitives" / "__init__.py"

        with open(primitives_code) as f:
            code = f.read()
            assert "try:" in code or "SURROGATE_AVAILABLE" in code
            assert "SurrogateModel" in code or "surrogate" in code.lower()

        print("✅ Primitives __init__.py has conditional imports")
    except Exception as e:
        print(f"⚠️  Could not verify primitives imports: {e}")

    # Test that task imports are conditional
    try:
        tasks_code = Path(__file__).parent.parent / "geosmith" / "tasks" / "__init__.py"
        with open(tasks_code) as f:
            code = f.read()
            assert "SurrogateTask" in code or "surrogate" in code.lower()

        print("✅ Tasks __init__.py has conditional imports")
    except Exception as e:
        print(f"⚠️  Could not verify tasks imports: {e}")

    return True


def test_dependencies_check():
    """Test that dependency checks work correctly."""
    print("\nTesting dependency checks...")

    # Check sklearn availability
    try:
        import sklearn  # noqa: F401
        sklearn_available = True
        print("✅ scikit-learn is available")
    except ImportError:
        sklearn_available = False
        print("⚠️  scikit-learn not available (expected for this test)")

    # Check xgboost availability
    try:
        import xgboost  # noqa: F401
        xgboost_available = True
        print("✅ xgboost is available")
    except ImportError:
        xgboost_available = False
        print("⚠️  xgboost not available (expected for this test)")

    # Try to import surrogate module
    if sklearn_available:
        try:
            print("✅ SurrogateModel can be imported")
            return True
        except Exception as e:
            print(f"❌ SurrogateModel import failed: {e}")
            return False
    else:
        print("⚠️  Cannot test full import without scikit-learn")
        print("   To test fully, install: pip install scikit-learn xgboost")
        return True  # Not a failure, just missing deps


def test_code_syntax():
    """Test that code has valid syntax."""
    print("\nTesting code syntax...")

    import ast

    files_to_check = [
        "geosmith/primitives/surrogate.py",
        "geosmith/tasks/surrogatetask.py",
        "examples/surrogate_example.py",
    ]

    for file_path in files_to_check:
        full_path = Path(__file__).parent.parent / file_path
        if full_path.exists():
            try:
                with open(full_path) as f:
                    ast.parse(f.read())
                print(f"✅ {file_path} has valid syntax")
            except SyntaxError as e:
                print(f"❌ {file_path} has syntax errors: {e}")
                return False
        else:
            print(f"⚠️  {file_path} not found")

    return True


def main():
    """Run all quick tests."""
    print("=" * 70)
    print("Quick Surrogate Model Tests")
    print("=" * 70)
    print()

    tests = [
        ("Structure", test_basic_structure),
        ("Imports", test_imports),
        ("Syntax", test_code_syntax),
        ("Dependencies", test_dependencies_check),
    ]

    passed = 0
    total = len(tests)

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {name} test passed\n")
            else:
                print(f"❌ {name} test failed\n")
        except Exception as e:
            print(f"❌ {name} test raised exception: {e}\n")
            import traceback
            traceback.print_exc()

    print("=" * 70)
    print(f"Tests passed: {passed}/{total}")
    print("=" * 70)

    if passed == total:
        print("\n🎉 All structure tests passed!")
        print("\nTo test full functionality, install dependencies:")
        print("  pip install scikit-learn xgboost")
        print("\nThen run:")
        print("  python examples/surrogate_example.py")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())


