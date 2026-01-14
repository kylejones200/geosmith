"""Helper for optional dependency imports with clean fallbacks.

This module provides utilities to handle optional dependencies consistently
across the codebase, eliminating repetitive try/except ImportError patterns.
"""

from typing import Any


def optional_import(
    module_path: str,
    names: list[str],
    module_name: str | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Import optional dependencies with clean fallback.

    Args:
        module_path: Full import path (e.g., 'geosmith.workflows.dlis')
        names: List of names to import from the module
        module_name: Optional module name for AVAILABLE flag (defaults to last part of path)

    Returns:
        Tuple of (available: bool, imports: dict[str, Any])
        - available: True if import succeeded, False otherwise
        - imports: Dictionary mapping name -> imported object (or None if import failed)

    Example:
        >>> available, imports = optional_import(
        ...     'geosmith.workflows.dlis',
        ...     ['DlisParser', 'read_dlis_file'],
        ...     'DLIS'
        ... )
        >>> DLIS_AVAILABLE = available
        >>> DlisParser = imports['DlisParser']
        >>> read_dlis_file = imports['read_dlis_file']
    """
    try:
        module = __import__(module_path, fromlist=names, level=0)
        result = {name: getattr(module, name) for name in names}
        return True, result
    except ImportError:
        result = {name: None for name in names}  # type: ignore
        return False, result


def optional_import_single(
    module_path: str,
    name: str,
) -> tuple[bool, Any]:
    """Import a single optional dependency.

    Convenience function for importing a single name.

    Args:
        module_path: Full import path
        name: Name to import

    Returns:
        Tuple of (available: bool, imported_object: Any)

    Example:
        >>> available, sklearn = optional_import_single('sklearn.ensemble', 'RandomForestClassifier')
        >>> if available:
        ...     model = sklearn()
    """
    available, imports = optional_import(module_path, [name])
    return available, imports[name]

