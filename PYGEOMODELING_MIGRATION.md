# PyGeomodeling to GeoSmith Migration Plan

This document outlines the migration of pygeomodeling features into GeoSmith's 4-layer architecture.

## Feature Mapping

### Layer 1: Objects (Data Representations)

**Current pygeomodeling**: Mixed data structures
**Target GeoSmith**: Immutable Layer 1 objects

| PyGeomodeling | GeoSmith Layer 1 | Status |
|---------------|------------------|--------|
| Well data (well_data.py) | `PointSet` with 3D coordinates + attributes | 📋 To migrate |
| Reservoir grid (reservoir_formats.py) | `RasterGrid` (3D) | 📋 To migrate |
| Formation tops | `PointSet` with depth attributes | 📋 To migrate |
| Facies data | `PointSet` with categorical attributes | 📋 To migrate |

### Layer 2: Primitives (Pure Operations)

**Current pygeomodeling**: Mixed with I/O and plotting
**Target GeoSmith**: Pure operations, no I/O

| PyGeomodeling | GeoSmith Layer 2 | Status |
|---------------|------------------|--------|
| `kriging.py` - OrdinaryKriging, UniversalKriging | `geosmith.primitives.kriging` | ✅ Migrated |
| `variogram.py` - VariogramModel, compute_experimental_variogram | `geosmith.primitives.variogram` | ✅ Migrated |
| `model_gp.py` - GP models | `geosmith.primitives.models` (BaseSpatialModel) | 📋 To migrate |
| `advanced_kernels.py` | `geosmith.primitives.kernels` | 📋 To migrate |

### Layer 3: Tasks (User Intent Translation)

**Current pygeomodeling**: Mixed concerns
**Target GeoSmith**: Clean task interfaces

| PyGeomodeling | GeoSmith Layer 3 | Status |
|---------------|------------------|--------|
| `cross_validation.py` - SpatialKFold, BlockCV | `geosmith.tasks.crossvalidation` | 📋 To migrate |
| `unified_toolkit.py` - UnifiedSPE9Toolkit | `geosmith.tasks.modelingtask` | 📋 To migrate |
| `well_log_processor.py` | `geosmith.tasks.welltask` | 📋 To migrate |
| `facies.py` - FaciesClassifier | `geosmith.tasks.faciestask` | 📋 To migrate |
| `formation_tops.py` | `geosmith.tasks.formationtask` | 📋 To migrate |

### Layer 4: Workflows (Public API + I/O)

**Current pygeomodeling**: Mixed I/O and workflows
**Target GeoSmith**: Clean separation

| PyGeomodeling | GeoSmith Layer 4 | Status |
|---------------|------------------|--------|
| `grdecl_parser.py` - GRDECLParser | `geosmith.workflows.grdecl.read_grdecl` | ✅ Migrated |
| `plot.py` - SPE9Plotter | `geosmith.workflows.plot` (enhanced) | 📋 To migrate |
| `serialization.py` | `geosmith.workflows.io` (model save/load) | 📋 To migrate |
| `reservoir_formats.py` | `geosmith.workflows.io` (reservoir formats) | 📋 To migrate |

## Migration Priority

### Phase 1: Core Geostatistics (High Priority)
1. ✅ Kriging → Layer 2 primitives (`geosmith.primitives.kriging`)
2. ✅ Variogram analysis → Layer 2 primitives (`geosmith.primitives.variogram`)
3. 📋 GP models → Layer 2 primitives (BaseSpatialModel)

### Phase 2: I/O and Data (High Priority)
4. ✅ GRDECL parser → Layer 4 workflows (`geosmith.workflows.grdecl`)
5. 📋 Reservoir grid formats → Layer 1 objects + Layer 4 workflows
6. 📋 Well data → Layer 1 objects (PointSet)

### Phase 3: Modeling Workflows (Medium Priority)
7. ✅ Cross-validation → Layer 3 tasks
8. ✅ Unified toolkit → Layer 3 tasks
9. ✅ Model serialization → Layer 4 workflows

### Phase 4: Domain-Specific (Lower Priority)
10. ✅ Facies classification → Layer 3 tasks
11. ✅ Formation tops → Layer 3 tasks
12. ✅ Well log processing → Layer 3 tasks

## Migration Strategy

1. **Start with Primitives**: Migrate kriging and variogram first (pure operations)
2. **Then Objects**: Create reservoir grid objects (Layer 1)
3. **Then Tasks**: Wrap operations in tasks (Layer 3)
4. **Finally Workflows**: Add I/O and public API (Layer 4)
5. **Compatibility Layer**: Create deprecation shims for old pygeomodeling API

## Compatibility

A compatibility layer will be created to allow gradual migration:

```python
# geosmith.compat.pygeomodeling
from geosmith.compat.pygeomodeling import (
    GRDECLParser,  # Wraps new implementation
    OrdinaryKriging,  # Wraps new implementation
    UnifiedSPE9Toolkit,  # Wraps new implementation
)
```

This allows existing pygeomodeling code to work with minimal changes while migrating to the new architecture.

