"""Spatial cross-validation task.

Migrated from geosuite.ml.cross_validation.
Layer 3: Tasks - User intent translation.
"""

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd

# Optional scikit-learn dependency
try:
    from sklearn.model_selection import BaseCrossValidator

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Create a dummy BaseCrossValidator for type hints
    from abc import ABC

    class BaseCrossValidator(ABC):  # type: ignore
        """Placeholder for BaseCrossValidator when sklearn is not available."""

        pass

logger = logging.getLogger(__name__)


class WellBasedKFold(BaseCrossValidator):
    """K-Fold cross-validation that groups by well to prevent data leakage.

    This ensures that data from the same well stays together in train/test splits,
    preventing overfitting due to spatial correlation within wells.

    Example:
        >>> from geosmith.tasks import WellBasedKFold
        >>> from sklearn.model_selection import cross_val_score
        >>>
        >>> cv = WellBasedKFold(n_splits=5, well_col='WELL')
        >>> scores = cross_val_score(model, X, y, cv=cv)
    """

    def __init__(
        self,
        n_splits: int = 5,
        well_col: str = "WELL",
        shuffle: bool = False,
        random_state: Optional[int] = None,
    ):
        """Initialize well-based K-Fold cross-validator.

        Args:
            n_splits: Number of folds.
            well_col: Column name containing well identifiers.
            shuffle: Whether to shuffle wells before splitting.
            random_state: Random seed for shuffling.
        """
        self.n_splits = n_splits
        self.well_col = well_col
        self.shuffle = shuffle
        self.random_state = random_state

    def split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        groups: Optional[np.ndarray] = None,
    ):
        """Generate indices to split data into training and test sets.

        Args:
            X: Input data (must be DataFrame with well_col).
            y: Target values (unused).
            groups: Group labels (unused, well_col is used instead).

        Yields:
            train_indices: Training set indices.
            test_indices: Test set indices.

        Raises:
            ValueError: If X is not a DataFrame or well_col is missing.
        """
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            raise ValueError(
                f"X must be a DataFrame with '{self.well_col}' column"
            )

        df = X.copy()

        if self.well_col not in df.columns:
            raise ValueError(
                f"Column '{self.well_col}' not found in X. "
                f"Available columns: {list(df.columns)}"
            )

        # Get unique wells
        unique_wells = df[self.well_col].unique()
        n_wells = len(unique_wells)

        if n_wells < self.n_splits:
            raise ValueError(
                f"Number of unique wells ({n_wells}) must be >= n_splits ({self.n_splits})"
            )

        # Shuffle wells if requested
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            unique_wells = rng.permutation(unique_wells)

        # Split wells into folds
        fold_size = n_wells // self.n_splits
        remainder = n_wells % self.n_splits

        start = 0
        for fold in range(self.n_splits):
            # Adjust fold size for remainder
            end = start + fold_size + (1 if fold < remainder else 0)

            # Get test wells for this fold
            test_wells = unique_wells[start:end]

            # Get indices
            test_mask = df[self.well_col].isin(test_wells)
            train_mask = ~test_mask

            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]

            logger.debug(
                f"Fold {fold + 1}/{self.n_splits}: "
                f"{len(train_indices)} train, {len(test_indices)} test samples, "
                f"{len(test_wells)} test wells"
            )

            yield train_indices, test_indices

            start = end

    def get_n_splits(
        self,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        groups: Optional[np.ndarray] = None,
    ) -> int:
        """Returns the number of splitting iterations in the cross-validator.

        Args:
            X: Input data (unused).
            y: Target values (unused).
            groups: Group labels (unused).

        Returns:
            Number of splits.
        """
        return self.n_splits


class SpatialKFold(BaseCrossValidator):
    """Spatial K-Fold cross-validation based on coordinates.

    Groups data by spatial proximity to prevent data leakage from spatial correlation.

    Example:
        >>> from geosmith.tasks import SpatialKFold
        >>> from geosmith import PointSet
        >>>
        >>> points = PointSet(coordinates=coords)
        >>> cv = SpatialKFold(n_splits=5, coordinates=points.coordinates)
        >>> scores = cross_val_score(model, X, y, cv=cv)
    """

    def __init__(
        self,
        n_splits: int = 5,
        coordinates: Optional[np.ndarray] = None,
        shuffle: bool = False,
        random_state: Optional[int] = None,
    ):
        """Initialize spatial K-Fold cross-validator.

        Args:
            n_splits: Number of folds.
            coordinates: Spatial coordinates (n_samples, n_dims).
            shuffle: Whether to shuffle before splitting.
            random_state: Random seed for shuffling.
        """
        self.n_splits = n_splits
        self.coordinates = coordinates
        self.shuffle = shuffle
        self.random_state = random_state

    def split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        groups: Optional[np.ndarray] = None,
    ):
        """Generate indices to split data into training and test sets.

        Args:
            X: Input data.
            y: Target values (unused).
            groups: Group labels (unused).

        Yields:
            train_indices: Training set indices.
            test_indices: Test set indices.

        Raises:
            ValueError: If coordinates are not provided.
        """
        if self.coordinates is None:
            raise ValueError("coordinates must be provided to SpatialKFold")

        n_samples = len(X)
        indices = np.arange(n_samples)

        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            indices = rng.permutation(indices)

        # Simple spatial split: divide into n_splits groups
        fold_size = n_samples // self.n_splits
        remainder = n_samples % self.n_splits

        start = 0
        for fold in range(self.n_splits):
            end = start + fold_size + (1 if fold < remainder else 0)

            test_indices = indices[start:end]
            train_indices = np.concatenate([indices[:start], indices[end:]])

            yield train_indices, test_indices

            start = end

    def get_n_splits(
        self,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        groups: Optional[np.ndarray] = None,
    ) -> int:
        """Returns the number of splitting iterations.

        Args:
            X: Input data (unused).
            y: Target values (unused).
            groups: Group labels (unused).

        Returns:
            Number of splits.
        """
        return self.n_splits

