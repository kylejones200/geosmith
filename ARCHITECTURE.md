# GeoSmith 4-Layer Architecture

## Overview

GeoSmith implements a strict 4-layer architecture with hard boundaries between layers. Imports flow one way only, ensuring clean separation of concerns.

## Layer Structure

### Layer 1: Objects (`geosmith.objects`)
**Purpose**: Immutable data representations

**Dependencies**: Python standard library + numpy + pandas only
- ❌ NO geopandas, shapely, rasterio, xarray, pyproj, matplotlib
- ❌ NO I/O libraries

**Components**:
- `GeoIndex`: CRS, bounds, axis order
- `PointSet`: Point clouds with coordinates and optional attributes
- `LineSet`: Lines with ordered vertices and optional attributes
- `PolygonSet`: Polygons with rings and optional attributes
- `RasterGrid`: Gridded values with affine transform, resolution, nodata
- `GeoTable`: Pandas DataFrame wrapper with geometry column

**Validators**: All objects validate coordinate shapes, index alignment, bounds validity, and CRS presence.

### Layer 2: Primitives (`geosmith.primitives`)
**Purpose**: Algorithm interfaces and pure operations

**Dependencies**: numpy, pandas + optional shapely/pyproj (behind adapters)
- ❌ NO file I/O
- ❌ NO plotting

**Components**:
- `BaseObject`, `BaseEstimator`, `BaseTransformer`, `BaseSpatialModel`, `BaseRasterModel`
- Tag system: `supports_crs_transform`, `requires_projected_crs`, `supports_3d`, etc.
- Geometry operations: distance metrics, nearest neighbor, point in polygon, line length, polygon area, bounding boxes
- Raster operations: grid resample, zonal reduce

**Adapters**: Optional backends (shapely, pyproj) behind small adapter functions.

### Layer 3: Tasks (`geosmith.tasks`)
**Purpose**: User intent translation

**Dependencies**: Can import geopandas/rasterio if present (optional, isolated)
- ❌ NO matplotlib
- ✅ Optional geopandas/rasterio with clean fallbacks

**Components**:
- `FeatureTask`: Vector operations (buffer, spatial join, distance to nearest)
- `RasterTask`: Raster operations (reproject, clip, resample, zonal stats)
- `RouteTask`: Placeholder for routing
- `ChangeTask`: Placeholder for change detection

**Input/Output**: Accepts user formats (pandas, geopandas, shapely, xarray, rasterio), converts to Layer 1 objects, returns Layer 1 objects or DataFrames.

### Layer 4: Workflows (`geosmith.workflows`)
**Purpose**: Public entry points

**Dependencies**: Can import I/O and plotting libraries
- ✅ File loading/saving (geopandas, rasterio)
- ✅ Plotting (matplotlib)

**Components**:
- `read_vector`, `write_vector`: Vector I/O
- `read_raster`, `write_raster`: Raster I/O
- `make_features`: Vector feature pipeline
- `process_raster`: Raster processing pipeline
- `zonal_stats`: Zonal statistics workflow
- `reproject_to`: Reprojection workflow
- Plot helpers: `plot_points`, `plot_polygons`, `plot_raster`

## Import Rules

1. **Objects** → Nothing (only stdlib + numpy + pandas)
2. **Primitives** → Objects only
3. **Tasks** → Primitives + Objects (optional geopandas/rasterio adapters)
4. **Workflows** → Tasks + Primitives + Objects + I/O libraries

## Public API

Only workflows and base types are exposed in `geosmith.__init__`:

```python
from geosmith import (
    # Objects
    PointSet, LineSet, PolygonSet, RasterGrid, GeoIndex, GeoTable,
    # Base classes
    BaseObject, BaseEstimator, BaseTransformer, BaseSpatialModel, BaseRasterModel,
    # Workflows
    make_features, process_raster, zonal_stats, reproject_to,
    read_vector, write_vector, read_raster, write_raster,
)
```

## Migration Status

### ✅ Completed Features

1. **Vector Feature Pipeline**:
   - Buffer points/polygons
   - Spatial join (points in polygons)
   - Distance to nearest neighbors

2. **Raster Processing Pipeline**:
   - Clip raster to polygon
   - Resample raster to new resolution
   - Zonal statistics

### 📋 Next Migrations (Priority Order)

1. Advanced spatial joins (left, right, outer)
2. Raster reprojection with proper CRS transformation
3. Raster band math operations
4. Vector overlay operations (union, intersection, difference)
5. Raster-to-vector conversion
6. Vector-to-raster conversion
7. Spatial indexing (R-tree, quadtree)
8. Line simplification and smoothing
9. Polygon simplification
10. Raster mosaic/merge

