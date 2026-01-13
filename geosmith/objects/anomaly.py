"""Anomaly detection objects compatible with AnomSmith.

These objects are compatible with AnomSmith's detection interfaces,
allowing seamless integration without conversion when AnomSmith is available.
AnomSmith is NOT a hard dependency - these objects work independently.
"""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd

from geosmith.objects.pointset import PointSet


@dataclass(frozen=True)
class AnomalyScores:
    """Anomaly scores compatible with AnomSmith's scoring interface.

    Represents anomaly scores for spatial data points, compatible with
    AnomSmith's detection results for seamless integration.

    Attributes:
        scores: Array of anomaly scores (n_points,). Higher scores indicate anomalies.
        points: PointSet that the scores correspond to.
        threshold: Optional threshold for binary classification.
        method: Optional method name used for detection.
    """

    scores: np.ndarray
    points: PointSet
    threshold: Optional[float] = None
    method: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate AnomalyScores parameters."""
        if not isinstance(self.scores, np.ndarray):
            raise ValueError(f"scores must be numpy array, got {type(self.scores)}")

        if len(self.scores) != len(self.points.coordinates):
            raise ValueError(
                f"scores length ({len(self.scores)}) must match "
                f"number of points ({len(self.points.coordinates)})"
            )

        if self.threshold is not None and not isinstance(self.threshold, (int, float)):
            raise ValueError(f"threshold must be numeric, got {type(self.threshold)}")

    def to_anomalies(self, threshold: Optional[float] = None) -> np.ndarray:
        """Convert scores to binary anomaly labels.

        Args:
            threshold: Optional threshold. Uses self.threshold if not provided.

        Returns:
            Boolean array where True indicates anomaly.
        """
        thresh = threshold if threshold is not None else self.threshold
        if thresh is None:
            raise ValueError("threshold must be provided or set in AnomalyScores")

        return self.scores > thresh

    def to_anomsmith(self):
        """Convert to AnomSmith format (if AnomSmith is available).

        Returns:
            AnomSmith-compatible object, or self if AnomSmith not available.
        """
        try:
            from anomsmith.objects import AnomalyScores as ASAnomalyScores

            return ASAnomalyScores(
                scores=self.scores,
                data=self.points.coordinates,
                threshold=self.threshold,
                method=self.method,
            )
        except ImportError:
            return self

    @classmethod
    def from_anomsmith(cls, as_scores):
        """Create from AnomSmith AnomalyScores object.

        Args:
            as_scores: AnomSmith AnomalyScores object.

        Returns:
            GeoSmith AnomalyScores object.
        """
        # Convert AnomSmith data back to PointSet
        points = PointSet(coordinates=as_scores.data)
        return cls(
            scores=as_scores.scores,
            points=points,
            threshold=as_scores.threshold,
            method=as_scores.method,
        )

    def __repr__(self) -> str:
        """String representation."""
        n_points = len(self.scores)
        n_anomalies = (
            np.sum(self.to_anomalies()) if self.threshold is not None else None
        )
        method_str = f", method='{self.method}'" if self.method else ""
        anomaly_str = f", n_anomalies={n_anomalies}" if n_anomalies is not None else ""
        return f"AnomalyScores(n_points={n_points}{anomaly_str}{method_str})"


@dataclass(frozen=True)
class SpatialAnomalyResult:
    """Spatial anomaly detection result.

    Contains anomaly scores and optional spatial context for geospatial
    anomaly detection. Compatible with AnomSmith's detection interfaces.

    Attributes:
        scores: AnomalyScores object with detection results.
        spatial_features: Optional DataFrame with spatial features used for detection.
        metadata: Optional dictionary with detection metadata.
    """

    scores: AnomalyScores
    spatial_features: Optional[pd.DataFrame] = None
    metadata: Optional[dict] = None

    def __post_init__(self) -> None:
        """Validate SpatialAnomalyResult parameters."""
        if not isinstance(self.scores, AnomalyScores):
            raise ValueError(f"scores must be AnomalyScores, got {type(self.scores)}")

        if self.spatial_features is not None:
            if not isinstance(self.spatial_features, pd.DataFrame):
                raise ValueError(
                    f"spatial_features must be pandas DataFrame, "
                    f"got {type(self.spatial_features)}"
                )
            if len(self.spatial_features) != len(self.scores.points.coordinates):
                raise ValueError(
                    f"spatial_features length ({len(self.spatial_features)}) "
                    f"must match number of points "
                    f"({len(self.scores.points.coordinates)})"
                )

    def to_anomsmith(self):
        """Convert to AnomSmith format (if AnomSmith is available).

        Returns:
            AnomSmith-compatible object, or self if AnomSmith not available.
        """
        try:
            from anomsmith.objects import AnomalyResult as ASAnomalyResult

            as_scores = self.scores.to_anomsmith()
            return ASAnomalyResult(
                scores=as_scores,
                features=self.spatial_features,
                metadata=self.metadata,
            )
        except ImportError:
            return self

    def __repr__(self) -> str:
        """String representation."""
        has_features = self.spatial_features is not None
        has_metadata = self.metadata is not None
        return (
            f"SpatialAnomalyResult(scores={self.scores}, "
            f"has_features={has_features}, has_metadata={has_metadata})"
        )
