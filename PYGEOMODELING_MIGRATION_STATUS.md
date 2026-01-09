# PyGeomodeling Migration Status

## Summary

This document tracks the migration of pygeomodeling features into GeoSmith's 4-layer architecture.

## Completed Migrations ✅

### 1. Variogram Analysis (Layer 2: Primitives)
**Location**: `geosmith.primitives.variogram`

**Migrated Features**:
- `VariogramModel` - Immutable dataclass for variogram parameters
- `compute_experimental_variogram` - Compute experimental variogram from PointSet
- `fit_variogram_model` - Fit theoretical models (spherical, exponential, gaussian, linear)
- `predict_variogram` - Predict variogram values at distances

**Key Changes**:
- Now works with `PointSet` objects instead of raw numpy arrays
- Pure operations (no I/O, no plotting)
- Numba-accelerated for performance
- Full test coverage

**Usage**:
```python
from geosmith import PointSet
from geosmith.primitives.variogram import (
    compute_experimental_variogram,
    fit_variogram_model,
)

# Create sample points
points = PointSet(coordinates=sample_coords)
values = sample_values

# Compute experimental variogram
lags, semi_vars, n_pairs = compute_experimental_variogram(points, values)

# Fit model
model = fit_variogram_model(lags, semi_vars, model_type="spherical")
```

### 2. Kriging (Layer 2: Primitives)
**Location**: `geosmith.primitives.kriging`

**Migrated Features**:
- `OrdinaryKriging` - Ordinary kriging interpolation (extends `BaseSpatialModel`)
- `KrigingResult` - Container for predictions and variance

**Key Changes**:
- Implements `BaseSpatialModel` interface
- Works with `PointSet` objects
- Returns `KrigingResult` with predictions and variance
- Numba-accelerated distance calculations
- Full test coverage

**Usage**:
```python
from geosmith import PointSet
from geosmith.primitives.kriging import OrdinaryKriging
from geosmith.primitives.variogram import VariogramModel

# Create variogram model
variogram = VariogramModel(
    model_type="spherical",
    nugget=0.1,
    sill=2.0,
    range_param=5.0,
    partial_sill=1.9,
    r_squared=0.95,
)

# Fit kriging
samples = PointSet(coordinates=sample_coords)
kriging = OrdinaryKriging(variogram_model=variogram)
kriging.fit(samples, sample_values)

# Predict
queries = PointSet(coordinates=query_coords)
result = kriging.predict(queries, return_variance=True)
print(f"Predictions: {result.predictions}")
print(f"Variance: {result.variance}")
```

### 3. GRDECL Parser (Layer 4: Workflows)
**Location**: `geosmith.workflows.grdecl`

**Migrated Features**:
- `read_grdecl` - Read GRDECL files and extract properties
- `write_grdecl` - Write GRDECL files from reservoir data

**Key Changes**:
- Clean I/O interface in Layer 4
- Can return dictionary or `RasterGrid` object
- Proper error handling and logging

**Usage**:
```python
from geosmith.workflows.grdecl import read_grdecl

# Read all properties
data = read_grdecl('SPE9.GRDECL')
print(f"Grid: {data['dimensions']}")
print(f"Properties: {list(data['properties'].keys())}")

# Read specific property as RasterGrid
permx = read_grdecl('SPE9.GRDECL', property_name='PERMX')
```

### 4. Compatibility Layer
**Location**: `geosmith.compat.pygeomodeling`

**Features**:
- Drop-in replacements for pygeomodeling classes
- Deprecation warnings guide users to new API
- Allows gradual migration

**Available Shims**:
- `GRDECLParser` - Wraps `read_grdecl`
- `OrdinaryKriging` - Wraps GeoSmith's `OrdinaryKriging`
- `VariogramModel` - Wraps GeoSmith's `VariogramModel`
- `compute_experimental_variogram` - Wraps GeoSmith's function
- `fit_variogram_model` - Wraps GeoSmith's function

**Usage**:
```python
# Old pygeomodeling code still works (with deprecation warning)
from geosmith.compat.pygeomodeling import GRDECLParser, OrdinaryKriging

parser = GRDECLParser('SPE9.GRDECL')
data = parser.load_data()
```

## In Progress 📋

### GP Models (Layer 2: Primitives)
- Need to migrate `model_gp.py` to extend `BaseSpatialModel`
- Support for scikit-learn and GPyTorch backends
- Advanced kernels

### Cross-Validation (Layer 3: Tasks)
- Spatial K-Fold
- Block CV
- Hyperparameter tuning

### Well Data Processing (Layer 1: Objects + Layer 4: Workflows)
- LAS file parsing
- Well log upscaling
- Formation tops detection

## Migration Guide

### For Users

1. **Update imports**:
   ```python
   # Old
   from pygeomodeling import OrdinaryKriging, VariogramModel
   
   # New
   from geosmith.primitives.kriging import OrdinaryKriging
   from geosmith.primitives.variogram import VariogramModel
   ```

2. **Use PointSet objects**:
   ```python
   # Old
   kriging.fit(coordinates, values)
   
   # New
   from geosmith import PointSet
   points = PointSet(coordinates=coordinates)
   kriging.fit(points, values)
   ```

3. **Use new I/O functions**:
   ```python
   # Old
   from pygeomodeling import GRDECLParser
   parser = GRDECLParser('file.GRDECL')
   data = parser.load_data()
   
   # New
   from geosmith.workflows.grdecl import read_grdecl
   data = read_grdecl('file.GRDECL')
   ```

### For Developers

1. **Follow 4-layer architecture**:
   - Layer 1: Immutable objects (PointSet, RasterGrid, etc.)
   - Layer 2: Pure primitives (no I/O, no plotting)
   - Layer 3: Tasks (user intent translation)
   - Layer 4: Workflows (I/O, plotting, public API)

2. **Add tests**:
   - Unit tests for primitives
   - Integration tests for workflows
   - Compatibility tests for shims

3. **Update documentation**:
   - Migration examples
   - API reference
   - Architecture guide

## Test Coverage

- ✅ Variogram primitives: 7 tests passing
- ✅ Kriging primitives: 2 tests passing
- 📋 GRDECL parser: Tests needed
- 📋 Compatibility layer: Tests needed

## Next Steps

1. Add tests for GRDECL parser
2. Migrate GP models to Layer 2
3. Migrate cross-validation to Layer 3
4. Create comprehensive migration examples
5. Update documentation

