"""Example: GeoSmith and AnomSmith integration for spatial anomaly detection.

This example demonstrates how GeoSmith's spatial objects work seamlessly
with AnomSmith for anomaly detection in geospatial data.

AnomSmith is NOT a hard dependency - GeoSmith works independently.
"""

import numpy as np
import pandas as pd

from geosmith import PointSet, GeoIndex, AnomalyScores, SpatialAnomalyResult
from geosmith.objects._anomsmith_compat import (
    detect_spatial_anomalies,
    is_anomsmith_available,
)

# Set random seed for reproducibility
np.random.seed(42)

print("GeoSmith and AnomSmith Integration Example")
print("=" * 60)

# Create synthetic point data with some outliers
n_points = 100
normal_coords = np.random.randn(n_points - 5, 2) * 10
outlier_coords = np.random.randn(5, 2) * 30 + np.array([[50, 50], [-50, 50], [50, -50], [-50, -50], [0, 0]])
all_coords = np.vstack([normal_coords, outlier_coords])

# Create PointSet
index = GeoIndex(
    crs="EPSG:4326",
    bounds=(-100, -100, 100, 100),
)
points = PointSet(coordinates=all_coords, index=index)

print(f"\n1. Created PointSet with {len(points.coordinates)} points")
print(f"   Includes {len(outlier_coords)} spatial outliers")

# Try to detect anomalies using AnomSmith (if available)
if is_anomsmith_available():
    try:
        print("\n2. Detecting spatial anomalies using AnomSmith...")
        result = detect_spatial_anomalies(
            points,
            method="isolation_forest",
            threshold=0.5,
        )
        print(f"   ✓ Detected anomalies using AnomSmith")
        print(f"   Result: {result}")
        
        # Get binary anomaly labels
        anomalies = result.scores.to_anomalies()
        n_anomalies = np.sum(anomalies)
        print(f"   Found {n_anomalies} anomalies (threshold={result.scores.threshold})")
        
        # Show anomaly scores
        print(f"\n   Anomaly scores:")
        print(f"   Min: {result.scores.scores.min():.3f}")
        print(f"   Max: {result.scores.scores.max():.3f}")
        print(f"   Mean: {result.scores.scores.mean():.3f}")
        print(f"   Std: {result.scores.scores.std():.3f}")
        
    except ImportError as e:
        print(f"   ℹ AnomSmith not fully available: {e}")
        print(f"   Install with: pip install geosmith[anomsmith]")
else:
    print("\n2. AnomSmith not available")
    print("   Install with: pip install geosmith[anomsmith]")
    
    # Create manual anomaly scores as example
    print("\n   Creating example AnomalyScores manually...")
    # Simple distance-based anomaly score (farther from center = more anomalous)
    center = points.coordinates.mean(axis=0)
    distances = np.linalg.norm(points.coordinates - center, axis=1)
    scores = distances / distances.max()  # Normalize to 0-1
    
    anomaly_scores = AnomalyScores(
        scores=scores,
        points=points,
        threshold=0.7,
        method="distance_from_center",
    )
    
    result = SpatialAnomalyResult(scores=anomaly_scores)
    print(f"   Created: {result}")
    
    anomalies = anomaly_scores.to_anomalies()
    n_anomalies = np.sum(anomalies)
    print(f"   Found {n_anomalies} anomalies (threshold={anomaly_scores.threshold})")

print("\n" + "=" * 60)
print("Summary:")
print("- GeoSmith objects work independently (no AnomSmith required)")
print("- When AnomSmith is available, spatial anomaly detection works seamlessly")
print("- AnomalyScores and SpatialAnomalyResult are compatible with AnomSmith")
print("- Install AnomSmith: pip install geosmith[anomsmith]")

