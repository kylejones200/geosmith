# AnomSmith Compatibility

GeoSmith provides anomaly detection objects that are **compatible** with [AnomSmith's](https://github.com/kylejones200/anomsmith) detection interfaces, allowing seamless integration for spatial anomaly detection. **AnomSmith is NOT a hard dependency** - GeoSmith works independently.

## Compatible Objects

GeoSmith provides two anomaly detection objects that match AnomSmith's structure:

### `AnomalyScores`
Anomaly scores for spatial data points, compatible with AnomSmith's scoring interface.

```python
from geosmith import PointSet, AnomalyScores
import numpy as np

# Create points
points = PointSet(coordinates=np.array([[0, 0], [1, 1], [100, 100]]))

# Create anomaly scores
scores = np.array([0.1, 0.2, 0.9])  # Higher = more anomalous
anomaly_scores = AnomalyScores(
    scores=scores,
    points=points,
    threshold=0.5,
    method="isolation_forest"
)

# Convert to binary labels
anomalies = anomaly_scores.to_anomalies()  # [False, False, True]

# Use with AnomSmith (if available)
as_scores = anomaly_scores.to_anomsmith()  # Returns AnomSmith format if available
```

### `SpatialAnomalyResult`
Complete spatial anomaly detection result with scores and optional context.

```python
from geosmith import SpatialAnomalyResult, AnomalyScores, PointSet
import pandas as pd

# Create result
scores = AnomalyScores(scores=scores_array, points=points, threshold=0.5)
features = pd.DataFrame({"distance_to_center": distances})

result = SpatialAnomalyResult(
    scores=scores,
    spatial_features=features,
    metadata={"method": "isolation_forest", "n_anomalies": 5}
)

# Use with AnomSmith (if available)
as_result = result.to_anomsmith()  # Returns AnomSmith format if available
```

## Installation

### GeoSmith Only (No AnomSmith)
```bash
pip install geosmith
```

### GeoSmith with AnomSmith
```bash
pip install geosmith[anomsmith]
```

## Usage Patterns

### 1. GeoSmith Standalone
GeoSmith objects work independently without AnomSmith:

```python
from geosmith import PointSet, AnomalyScores
import numpy as np

points = PointSet(coordinates=coords)
scores = AnomalyScores(
    scores=np.random.rand(len(coords)),
    points=points,
    threshold=0.5
)
# Use scores directly - no AnomSmith needed
```

### 2. With AnomSmith (Optional)
When AnomSmith is available, use it for detection:

```python
from geosmith import PointSet
from geosmith.objects._anomsmith_compat import detect_spatial_anomalies

points = PointSet(coordinates=coords)

# Detect anomalies using AnomSmith (if available)
try:
    result = detect_spatial_anomalies(
        points,
        method="isolation_forest",
        threshold=0.5
    )
    anomalies = result.scores.to_anomalies()
    print(f"Found {np.sum(anomalies)} anomalies")
except ImportError:
    # AnomSmith not available - use manual detection
    pass
```

### 3. From AnomSmith
Convert AnomSmith results to GeoSmith:

```python
from geosmith.objects.anomaly import SpatialAnomalyResult
from geosmith.objects._anomsmith_compat import from_anomsmith_result
from anomsmith import detect_anomalies

# AnomSmith detection
as_result = detect_anomalies(data, method="isolation_forest")

# Convert to GeoSmith
gs_result = from_anomsmith_result(as_result)
```

## Compatibility Layer

The `geosmith.objects._anomsmith_compat` module provides helper functions:

```python
from geosmith.objects._anomsmith_compat import (
    is_anomsmith_available,
    detect_spatial_anomalies,
    to_anomsmith_scores,
    from_anomsmith_result,
)

# Check if AnomSmith is available
if is_anomsmith_available():
    # Detect spatial anomalies
    result = detect_spatial_anomalies(points, method="isolation_forest")
```

## Spatial Anomaly Detection

GeoSmith's spatial objects (PointSet) work seamlessly with AnomSmith for detecting:

- **Spatial outliers**: Points that are far from the main cluster
- **Density anomalies**: Points in unusually sparse or dense regions
- **Pattern anomalies**: Points that don't match expected spatial patterns
- **Attribute anomalies**: Points with unusual attribute values in spatial context

## Design Principles

1. **No Hard Dependency**: GeoSmith works without AnomSmith
2. **Same Structure**: Objects match AnomSmith's detection interfaces
3. **No Conversion**: Same underlying numpy/pandas structures - no data copying
4. **Seamless Integration**: When AnomSmith is available, objects work together
5. **Layer 1 Compliance**: Objects follow GeoSmith's Layer 1 rules (only stdlib + numpy + pandas)

## Example

See `examples/anomsmith_integration.py` for a complete example demonstrating:
- Creating GeoSmith PointSet objects
- Detecting anomalies with AnomSmith (when available)
- Using AnomalyScores independently when AnomSmith is not installed

## Benefits

- ✅ **No conversion overhead**: Same numpy/pandas structures
- ✅ **Optional dependency**: Install AnomSmith only when needed
- ✅ **Type safety**: Runtime validation matches AnomSmith's requirements
- ✅ **Seamless integration**: Works together when both are available
- ✅ **Independent operation**: GeoSmith works standalone
- ✅ **Spatial context**: Preserves spatial information (PointSet, CRS, bounds)

