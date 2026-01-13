"""Tests for anomaly detection objects compatible with AnomSmith."""

import numpy as np
import pytest

from geosmith import AnomalyScores, PointSet, SpatialAnomalyResult


class TestAnomalyScores:
    """Tests for AnomalyScores."""

    def test_valid_anomaly_scores(self):
        """Test creating valid AnomalyScores."""
        coords = np.array([[0, 0], [1, 1], [2, 2]])
        points = PointSet(coordinates=coords)
        scores = np.array([0.1, 0.2, 0.9])

        anomaly_scores = AnomalyScores(
            scores=scores, points=points, threshold=0.5, method="test"
        )
        assert len(anomaly_scores.scores) == 3
        assert anomaly_scores.threshold == 0.5
        assert anomaly_scores.method == "test"

    def test_mismatched_length(self):
        """Test that mismatched scores length raises error."""
        coords = np.array([[0, 0], [1, 1]])
        points = PointSet(coordinates=coords)
        scores = np.array([0.1, 0.2, 0.9])  # Wrong length

        with pytest.raises(ValueError, match="scores length"):
            AnomalyScores(scores=scores, points=points)

    def test_to_anomalies(self):
        """Test converting scores to binary labels."""
        coords = np.array([[0, 0], [1, 1], [2, 2]])
        points = PointSet(coordinates=coords)
        scores = np.array([0.1, 0.2, 0.9])

        anomaly_scores = AnomalyScores(
            scores=scores, points=points, threshold=0.5
        )
        anomalies = anomaly_scores.to_anomalies()
        assert anomalies.dtype == bool
        assert np.sum(anomalies) == 1  # Only one above threshold

    def test_to_anomsmith_no_dependency(self):
        """Test to_anomsmith when AnomSmith not available."""
        coords = np.array([[0, 0], [1, 1]])
        points = PointSet(coordinates=coords)
        scores = np.array([0.1, 0.9])

        anomaly_scores = AnomalyScores(scores=scores, points=points)
        result = anomaly_scores.to_anomsmith()
        # Should return self when AnomSmith not available
        assert result is anomaly_scores


class TestSpatialAnomalyResult:
    """Tests for SpatialAnomalyResult."""

    def test_valid_result(self):
        """Test creating valid SpatialAnomalyResult."""
        coords = np.array([[0, 0], [1, 1], [2, 2]])
        points = PointSet(coordinates=coords)
        scores = AnomalyScores(
            scores=np.array([0.1, 0.2, 0.9]), points=points, threshold=0.5
        )

        result = SpatialAnomalyResult(scores=scores)
        assert result.scores == scores

    def test_result_with_features(self):
        """Test SpatialAnomalyResult with spatial features."""
        import pandas as pd

        coords = np.array([[0, 0], [1, 1], [2, 2]])
        points = PointSet(coordinates=coords)
        scores = AnomalyScores(
            scores=np.array([0.1, 0.2, 0.9]), points=points, threshold=0.5
        )
        features = pd.DataFrame({"distance": [1.0, 1.4, 2.8]})

        result = SpatialAnomalyResult(scores=scores, spatial_features=features)
        assert len(result.spatial_features) == 3

    def test_mismatched_features_length(self):
        """Test that mismatched features length raises error."""
        import pandas as pd

        coords = np.array([[0, 0], [1, 1], [2, 2]])
        points = PointSet(coordinates=coords)
        scores = AnomalyScores(
            scores=np.array([0.1, 0.2, 0.9]), points=points, threshold=0.5
        )
        features = pd.DataFrame({"distance": [1.0, 1.4]})  # Wrong length

        with pytest.raises(ValueError, match="spatial_features length"):
            SpatialAnomalyResult(scores=scores, spatial_features=features)


