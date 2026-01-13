"""
Base classes for consistent API patterns in GeoSuite.

This module provides base classes following scikit-learn conventions
for transformers, estimators, and calculators used throughout GeoSuite.
"""

from .calculators import BaseCalculator
from .estimators import BaseEstimator
from .transformers import BaseTransformer

__all__ = [
    "BaseTransformer",
    "BaseCalculator",
    "BaseEstimator",
]
