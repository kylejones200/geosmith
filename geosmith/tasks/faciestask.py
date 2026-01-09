"""Facies classification task.

Migrated from geosuite.ml.classifiers.
Layer 3: Tasks - User intent translation.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from geosmith.objects.geotable import GeoTable

logger = logging.getLogger(__name__)

# Optional scikit-learn dependency
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    RandomForestClassifier = None  # type: ignore
    SVC = None  # type: ignore
    Pipeline = None  # type: ignore
    StandardScaler = None  # type: ignore


@dataclass
class FaciesResult:
    """Result from facies classification.

    Attributes:
        classes_: List of class names.
        y_pred: Predicted class labels.
        proba: Probability DataFrame (columns = classes_).
        model_name: Name of the model used.
        report: Classification report string (if test_size > 0).
    """

    classes_: List[str]
    y_pred: pd.Series
    proba: pd.DataFrame
    model_name: str
    report: str


class FaciesTask:
    """Task for facies classification.

    Works with GeoTable or Pandas DataFrame inputs.
    Returns GeoTable or DataFrame with predictions.
    """

    def __init__(self):
        """Initialize FaciesTask."""
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for facies classification. "
                "Install with: pip install geosmith[ml] or pip install scikit-learn"
            )

    def train_and_predict(
        self,
        data: GeoTable | pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        model_type: str = "SVM",
        test_size: float = 0.0,
        random_state: int = 42,
    ) -> FaciesResult:
        """Train a classifier and predict on the dataset.

        Args:
            data: GeoTable or DataFrame with features and target.
            feature_cols: List of feature column names.
            target_col: Name of target column.
            model_type: Model type ('SVM' or 'RF'/'RandomForest').
            test_size: Fraction of data to use for testing (0.0 = no split).
            random_state: Random seed.

        Returns:
            FaciesResult with predictions and probabilities.

        Raises:
            ImportError: If scikit-learn is not available.
            ValueError: If model_type is invalid.

        Example:
            >>> from geosmith.tasks import FaciesTask
            >>> from geosmith import GeoTable
            >>>
            >>> task = FaciesTask()
            >>> result = task.train_and_predict(
            ...     data=geotable,
            ...     feature_cols=['GR', 'NPHI', 'RHOB', 'PE'],
            ...     target_col='Facies',
            ...     model_type='RF'
            ... )
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for facies classification. "
                "Install with: pip install scikit-learn"
            )

        # Extract DataFrame if GeoTable
        if isinstance(data, GeoTable):
            df = data.data.copy()
        else:
            df = data.copy()

        X = df[feature_cols].copy()
        y = df[target_col].astype(str).copy()

        # Create model pipeline
        if model_type.upper() == "SVM":
            pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        SVC(
                            kernel="rbf",
                            C=10.0,
                            gamma="scale",
                            probability=True,
                            random_state=random_state,
                        ),
                    ),
                ]
            )
            model_name = "SVM (RBF)"
        elif model_type.upper() in ("RF", "RANDOMFOREST", "RANDOM_FOREST"):
            pipe = Pipeline(
                [
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=300,
                            max_depth=None,
                            random_state=random_state,
                        ),
                    )
                ]
            )
            model_name = "RandomForest"
        else:
            raise ValueError(
                f"Unsupported model_type: {model_type}. "
                "Use 'SVM' or 'RF'/'RandomForest'"
            )

        report = ""
        if test_size and test_size > 0:
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            pipe.fit(Xtr, ytr)
            yhat_te = pipe.predict(Xte)
            report = classification_report(yte, yhat_te)
            logger.info(f"Trained {model_name} with test_size={test_size}")
        else:
            pipe.fit(X, y)
            logger.info(f"Trained {model_name} on full dataset")

        # Predict on full dataset
        y_pred = pd.Series(pipe.predict(X), index=df.index, name="predicted")

        # Get probabilities
        if hasattr(pipe, "predict_proba"):
            P = pipe.predict_proba(X)
            classes = list(pipe.classes_)
        else:
            # Fallback
            classes = list(np.unique(y))
            P = np.zeros((len(X), len(classes)))
            class_to_idx = {c: i for i, c in enumerate(classes)}
            for i, c in enumerate(y_pred):
                P[i, class_to_idx[c]] = 1.0

        proba = pd.DataFrame(P, index=df.index, columns=classes)

        return FaciesResult(
            classes_=classes,
            y_pred=y_pred,
            proba=proba,
            model_name=model_name,
            report=report,
        )

