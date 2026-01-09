"""Compatibility layer for AnomSmith integration.

This module provides seamless integration between GeoSmith and AnomSmith
for anomaly detection when AnomSmith is available. AnomSmith is NOT a hard dependency.
"""

from typing import Optional, Union

import numpy as np

from geosmith.objects.anomaly import AnomalyScores, SpatialAnomalyResult
from geosmith.objects.pointset import PointSet


def is_anomsmith_available() -> bool:
    """Check if AnomSmith is available.

    Returns:
        True if AnomSmith can be imported, False otherwise.
    """
    try:
        import anomsmith  # noqa: F401
        return True
    except ImportError:
        return False


def detect_spatial_anomalies(
    points: PointSet,
    method: str = "isolation_forest",
    threshold: Optional[float] = None,
    **kwargs,
) -> SpatialAnomalyResult:
    """Detect spatial anomalies using AnomSmith if available.

    Args:
        points: PointSet to detect anomalies in.
        method: Detection method name.
        threshold: Optional threshold for binary classification.
        **kwargs: Additional arguments passed to AnomSmith detector.

    Returns:
        SpatialAnomalyResult with anomaly scores.

    Raises:
        ImportError: If AnomSmith is not available and method requires it.
    """
    if not is_anomsmith_available():
        raise ImportError(
            "AnomSmith is not available. Install with: pip install geosmith[anomsmith]"
        )

    try:
        from anomsmith import detect_anomalies

        # Convert PointSet to format AnomSmith expects
        data = points.coordinates

        # Detect anomalies using AnomSmith
        as_result = detect_anomalies(data, method=method, **kwargs)

        # Convert back to GeoSmith format
        scores = AnomalyScores(
            scores=as_result.scores,
            points=points,
            threshold=threshold or as_result.threshold,
            method=method,
        )

        return SpatialAnomalyResult(
            scores=scores,
            spatial_features=as_result.features if hasattr(as_result, "features") else None,
            metadata=as_result.metadata if hasattr(as_result, "metadata") else None,
        )
    except ImportError:
        raise ImportError(
            "AnomSmith is not available. Install with: pip install geosmith[anomsmith]"
        )


def to_anomsmith_scores(scores: Union[AnomalyScores, np.ndarray]) -> AnomalyScores:
    """Convert to AnomSmith-compatible AnomalyScores.

    Args:
        scores: GeoSmith AnomalyScores or numpy array of scores.

    Returns:
        AnomSmith-compatible AnomalyScores if available, otherwise GeoSmith AnomalyScores.
    """
    if isinstance(scores, AnomalyScores):
        return scores.to_anomsmith()
    else:
        raise ValueError(
            "scores must be AnomalyScores object. "
            "Create AnomalyScores first with points and scores."
        )


def from_anomsmith_result(as_result) -> SpatialAnomalyResult:
    """Convert from AnomSmith result to GeoSmith SpatialAnomalyResult.

    Args:
        as_result: AnomSmith anomaly detection result.

    Returns:
        GeoSmith SpatialAnomalyResult.
    """
    if not is_anomsmith_available():
        raise ImportError(
            "AnomSmith is not available. Install with: pip install geosmith[anomsmith]"
        )

    # Extract scores
    scores = AnomalyScores.from_anomsmith(as_result.scores)

    # Create result
    return SpatialAnomalyResult(
        scores=scores,
        spatial_features=as_result.features if hasattr(as_result, "features") else None,
        metadata=as_result.metadata if hasattr(as_result, "metadata") else None,
    )

