"""Numba JIT compilation helpers with graceful degradation.

This module provides Numba decorators with automatic fallback when Numba is not available.
Performance-critical numerical functions can use these decorators to achieve 10-100x speedups
with minimal code changes.

Migrated from geosuite.utils.numba_helpers.
Layer 2: Primitives - Pure operations.
"""

import os
import warnings
from typing import Callable, Optional

# Check if Numba should be disabled (for debugging)
NUMBA_DISABLE = os.environ.get("NUMBA_DISABLE_JIT", "0") == "1"

if NUMBA_DISABLE:
    warnings.warn("Numba JIT compilation disabled via NUMBA_DISABLE_JIT=1")
    NUMBA_AVAILABLE = False
    _njit = None
    _prange = None
else:
    try:
        from numba import njit as _njit, prange as _prange

        # Verify imports are actually callable
        if _njit is None or _prange is None:
            raise ImportError("Numba imports returned None")
        NUMBA_AVAILABLE = True
    except (ImportError, AttributeError, TypeError) as e:
        NUMBA_AVAILABLE = False
        _njit = None
        _prange = None
        warnings.warn(
            "Numba not available. Install for 10-100x speedups on numerical code: "
            "pip install numba>=0.58.0",
            ImportWarning,
        )


def njit(*args, **kwargs):
    """Numba JIT compilation decorator with graceful fallback.

    Args:
        cache (bool): Cache compiled functions for faster startup (recommended: True).
        parallel (bool): Enable automatic parallelization with prange.
        fastmath (bool): Enable fast math optimizations (may reduce precision slightly).

    Returns:
        Decorated function (compiled if Numba available, otherwise unchanged).

    Example:
        >>> from geosmith.primitives._numba_helpers import njit
        >>> import numpy as np
        >>>
        >>> @njit(cache=True)
        >>> def integrate(x, y):
        ...     result = 0.0
        ...     for i in range(1, len(x)):
        ...         result += (y[i] + y[i-1]) * (x[i] - x[i-1]) / 2
        ...     return result
    """
    # No-op decorator: return function unchanged
    def decorator(func: Callable) -> Callable:
        return func

    # Check if we can actually use numba
    if not NUMBA_AVAILABLE or _njit is None:
        # Handle both @njit and @njit() calling styles
        if len(args) == 1 and callable(args[0]) and not kwargs:
            # Called as @njit (no parentheses)
            return args[0]
        else:
            # Called as @njit(...) (with parentheses)
            return decorator
    else:
        # Numba is available and working, use real decorator
        return _njit(*args, **kwargs)


if NUMBA_AVAILABLE and _prange is not None:
    # Use real prange for parallel loops
    prange = _prange
else:
    # Fallback: prange becomes regular range
    prange = range  # type: ignore


__all__ = ["njit", "prange", "NUMBA_AVAILABLE"]

