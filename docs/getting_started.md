This guide walks through a complete GeoSmith workflow using synthetic data. The focus stays on structure and intent, not file formats.

We begin with a simple set of points and polygons represented in pandas.

```python
import pandas as pd
import numpy as np

points = pd.DataFrame(
    {
        "x": np.random.uniform(0, 100, size=50),
        "y": np.random.uniform(0, 100, size=50),
        "value": np.random.normal(10, 2, size=50),
    }
)

polygons = pd.DataFrame(
    {
        "xmin": [0, 50],
        "ymin": [0, 50],
        "xmax": [50, 100],
        "ymax": [50, 100],
        "region": ["A", "B"],
    }
)
```

GeoSmith tasks convert these inputs into spatial objects at the boundary.

```python
from geosmith.workflows import make_features

features = make_features(
    points=points,
    polygons=polygons,
    operation="point_in_polygon"
)
```

The result is a table of spatial features aligned to the original points.

```python
print(features.head())
```

This is the GeoSmith loop. Objects represent space. Primitives compute geometry. Tasks express intent. Workflows handle I O and integration.

