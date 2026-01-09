# GeoSuite to GeoSmith Migration Guide

This guide helps migrate code from GeoSuite to GeoSmith's 4-layer architecture.

## Architecture Differences

### GeoSuite Structure
- Flat module structure (geosuite.mining, geosuite.petro, etc.)
- Mixed concerns (I/O, computation, plotting in same modules)
- Direct dependencies on external libraries

### GeoSmith Structure
- **Layer 1 (Objects)**: Immutable data structures (PointSet, RasterGrid, etc.)
- **Layer 2 (Primitives)**: Pure operations (IDW, geometry ops, etc.)
- **Layer 3 (Tasks)**: User intent translation (FeatureTask, BlockModelTask, etc.)
- **Layer 4 (Workflows)**: Public API with I/O and plotting

## Migration Patterns

### 1. Point Data → PointSet

**GeoSuite:**
```python
import numpy as np
coords = np.array([[100, 200, 50], [150, 250, 60]])
values = np.array([2.5, 1.8])
```

**GeoSmith:**
```python
from geosmith import PointSet, GeoIndex
import numpy as np

coords = np.array([[100, 200, 50], [150, 250, 60]])
values = np.array([2.5, 1.8])

points = PointSet(
    coordinates=coords,
    attributes=pd.DataFrame({"grade": values}),
    index=GeoIndex(crs="EPSG:4326", bounds=(0, 0, 1000, 1000))
)
```

### 2. IDW Interpolation

**GeoSuite:**
```python
from geosuite.mining.interpolation import idw_interpolate

P = np.array([[100, 200, 50], [150, 250, 60]])
V = np.array([2.5, 1.8])
Q = np.array([[130, 220, 57]])

grade = idw_interpolate(P, V, Q, k=3, power=2.0)
```

**GeoSmith:**
```python
from geosmith import PointSet
from geosmith.primitives.interpolation import idw_interpolate

samples = PointSet(coordinates=P)
queries = PointSet(coordinates=Q)

grade = idw_interpolate(samples, V, queries, k=3, power=2.0)
```

### 3. Block Model Creation

**GeoSuite:**
```python
from geosuite.mining.block_model import create_block_model_grid
from geosuite.mining.interpolation import idw_interpolate

coords = samples[['x', 'y', 'z']].values
grid, info = create_block_model_grid(coords, block_size_xy=25, block_size_z=10)
grades = idw_interpolate(coords, values, grid, k=16)
```

**GeoSmith:**
```python
from geosmith import PointSet
from geosmith.tasks import BlockModelTask

samples = PointSet(coordinates=coords)
task = BlockModelTask()

# Create grid
grid_points, grid_info = task.create_block_model_grid(
    samples, block_size_xy=25, block_size_z=10
)

# Estimate grades
grades = task.estimate_grades(samples, values, grid_points, k=16)

# Or use convenience method
block_model = task.create_block_model(samples, values, block_size_xy=25, block_size_z=10)
```

### 4. Vector Operations

**GeoSuite:**
```python
# (GeoSuite may not have direct equivalents - use geopandas)
import geopandas as gpd
buffered = gdf.buffer(100)
```

**GeoSmith:**
```python
from geosmith import make_features, PointSet

points = PointSet(coordinates=coords)
buffered = make_features(
    points,
    operations={"buffer": {"distance": 100.0}}
)
```

### 5. Raster Operations

**GeoSuite:**
```python
# (GeoSuite may not have direct equivalents - use rasterio)
import rasterio
with rasterio.open('raster.tif') as src:
    data = src.read()
```

**GeoSmith:**
```python
from geosmith import read_raster, process_raster

raster = read_raster('raster.tif')
resampled = process_raster(
    raster,
    operations={"resample": {"target_shape": (100, 100), "target_transform": transform}}
)
```

## Feature Mapping

| GeoSuite Module | GeoSmith Location | Status |
|----------------|-------------------|--------|
| `mining.interpolation.idw_interpolate` | `primitives.interpolation.idw_interpolate` | ✅ Migrated |
| `mining.block_model.create_block_model_grid` | `tasks.BlockModelTask.create_block_model_grid` | ✅ Migrated |
| `mining.block_model.export_block_model` | `workflows.io.write_vector` (future) | 📋 Planned |
| `mining.drillhole.process_drillhole_data` | `workflows.io.read_vector` + helpers | 📋 Planned |
| `geospatial.*` | `tasks.FeatureTask`, `tasks.RasterTask` | ✅ Partial |
| `petro.*` | Not applicable (domain-specific) | ❌ Out of scope |
| `geomech.*` | Not applicable (domain-specific) | ❌ Out of scope |
| `ml.*` | Not applicable (ML-specific) | ❌ Out of scope |

## Migration Checklist

- [x] IDW interpolation → Layer 2 primitives
- [x] Block model grid creation → Layer 3 tasks
- [x] Block model grade estimation → Layer 3 tasks
- [ ] Block model export → Layer 4 workflows
- [ ] Drillhole data processing → Layer 1 objects + Layer 4 workflows
- [ ] Vector operations (buffer, join, etc.) → Layer 3 tasks
- [ ] Raster operations (clip, resample, etc.) → Layer 3 tasks
- [ ] I/O operations → Layer 4 workflows

## Compatibility Layer (Future)

A compatibility layer could be added to allow gradual migration:

```python
# geosmith.compat.geosuite
from geosmith.compat.geosuite import idw_interpolate  # Wraps new implementation
```

This would allow existing GeoSuite code to work with minimal changes while migrating to the new architecture.

## Next Steps

1. Identify which GeoSuite features you use
2. Map them to GeoSmith layers using this guide
3. Migrate one feature at a time
4. Test thoroughly
5. Update documentation

For questions or issues, see the [GeoSmith documentation](https://geosmith.readthedocs.io/).

