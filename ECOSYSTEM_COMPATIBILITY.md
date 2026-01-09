# *Smith Ecosystem Compatibility

GeoSmith is part of the *Smith ecosystem and follows strict compatibility rules to ensure clean dependency graphs and shared typing.

## Dependency Graph Rules

1. **timesmith is the foundation** - Everyone may depend on timesmith
2. **No cross-dependencies** - plotsmith, anomsmith, ressmith, geosmith must not depend on each other
3. **Optional heavy deps** - Only workflows may use optional heavy dependencies
4. **Core layers stay light** - Objects, primitives, and tasks must remain lightweight
5. **No downstream imports** - Never import a downstream repo from timesmith

## Shared Typing (Single Source of Truth)

**timesmith.typing** is the single source of truth for:
- `SeriesLike` - Time series data
- `PanelLike` - Panel data with entity keys
- `TableLike` - Time-aligned tables

### Import Pattern

```python
# ✅ CORRECT: Import from timesmith.typing
from timesmith.typing import SeriesLike, PanelLike, TableLike
from timesmith.typing.validators import (
    assert_series_like,
    assert_panel_like,
    assert_table_like,
)

# ❌ WRONG: Don't redefine locally
# class SeriesLike: ...  # NO!
```

### Validation Pattern

```python
# Use TimeSmith validators directly
from timesmith.typing.validators import assert_series_like

series = SeriesLike(data=pd.Series(...))
assert_series_like(series)  # Validates using TimeSmith's validators
```

## GeoSmith's 4-Layer Architecture

Each layer has strict import rules:

### Layer 1: Objects
- **Dependencies**: stdlib + numpy + pandas + timesmith (for typing)
- **Imports**: Nothing else
- **Time series**: Re-exports from `timesmith.typing`

### Layer 2: Primitives
- **Dependencies**: Objects + stdlib + numpy + pandas
- **Imports**: Objects only
- **No I/O, no plotting**

### Layer 3: Tasks
- **Dependencies**: Objects + Primitives + numpy + pandas
- **Imports**: Objects and Primitives only
- **No I/O, no plotting**

### Layer 4: Workflows
- **Dependencies**: All layers + optional heavy deps
- **Imports**: Anything needed for I/O and plotting
- **Can use**: geopandas, rasterio, matplotlib, plotsmith, etc.

## Versioning

GeoSmith pins timesmith as a minimum version:

```toml
dependencies = [
  "timesmith>=0.1.0",  # Required for typing
]
```

## Integration Testing

### Smoke Test

Run the smoke test to verify installation and basic functionality:

```bash
python scripts/smoke.py
```

### Integration Example

See `examples/timesmith_integration.py` for a complete example demonstrating:
- Importing from `timesmith.typing`
- Using `timesmith.typing.validators`
- Working with GeoSmith spatial objects

## No Circular Imports

GeoSmith ensures no circular imports by:
- Only importing from timesmith (upstream)
- Never importing from plotsmith, anomsmith, ressmith (peer repos)
- Using optional dependencies with try/except blocks

## Public API

GeoSmith's public API (`geosmith/__init__.py`) exports:
- Workflows (Layer 4)
- Base classes (Layer 2)
- Key objects (Layer 1, including time series types from timesmith.typing)

No deep imports or internal module re-exports.

## Compatibility Checklist

- ✅ timesmith is required dependency (not optional)
- ✅ SeriesLike, PanelLike, TableLike imported from timesmith.typing
- ✅ No local redefinition of time series types
- ✅ Validators imported from timesmith.typing.validators
- ✅ No circular imports
- ✅ No dependencies on peer repos (plotsmith, anomsmith, ressmith)
- ✅ Smoke test script exists
- ✅ Integration example exists
- ✅ All tests pass

