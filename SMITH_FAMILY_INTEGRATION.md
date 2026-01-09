# *Smith Family Integration

GeoSmith is designed to work seamlessly with other *Smith libraries while maintaining **no hard dependencies**. This document describes how GeoSmith integrates with PlotSmith and AnomSmith.

## Philosophy

- ✅ **Optional Integration**: All *Smith libraries are optional dependencies
- ✅ **Graceful Fallbacks**: GeoSmith works independently when other libraries aren't installed
- ✅ **Same Object Structure**: Compatible object types for seamless data exchange
- ✅ **No Conversion Overhead**: Direct compatibility when libraries are available

## PlotSmith Integration

**Repository**: https://github.com/kylejones200/plotsmith

### What PlotSmith Provides

PlotSmith provides publication-ready plotting utilities for ML models and data visualization.

### How GeoSmith Uses PlotSmith

GeoSmith's plotting functions (`geosmith.workflows.plot`) automatically detect and use PlotSmith when available:

```python
from geosmith import PointSet
from geosmith.workflows.plot import plot_points

points = PointSet(coordinates=coords)

# Automatically uses PlotSmith if available, falls back to matplotlib
ax = plot_points(points, use_plotsmith=None)  # Auto-detect

# Or explicitly request PlotSmith
ax = plot_points(points, use_plotsmith=True)  # Use PlotSmith if available

# Or force matplotlib
ax = plot_points(points, use_plotsmith=False)  # Use matplotlib
```

### Installation

```bash
# Install GeoSmith with PlotSmith support
pip install geosmith[plotsmith]

# Or install separately
pip install geosmith plotsmith
```

### Features

- **Auto-detection**: Plotting functions automatically detect PlotSmith
- **Graceful fallback**: Falls back to matplotlib if PlotSmith not available
- **Same API**: Plotting functions work the same way regardless of backend
- **Better plots**: When PlotSmith is available, plots are cleaner and publication-ready

### Current Integration

- ✅ `plot_points()` - Uses PlotSmith's `plot_scatter` when available
- 📋 `plot_polygons()` - Could use PlotSmith in future
- 📋 `plot_raster()` - Could use PlotSmith in future

## AnomSmith Integration

**Repository**: https://github.com/kylejones200/anomsmith

### What AnomSmith Provides

AnomSmith provides anomaly detection capabilities for time series and spatial data.

### How GeoSmith Uses AnomSmith

GeoSmith provides compatible anomaly detection objects that work seamlessly with AnomSmith:

```python
from geosmith import PointSet, AnomalyScores, SpatialAnomalyResult
from geosmith.objects._anomsmith_compat import (
    is_anomsmith_available,
    detect_spatial_anomalies,
)

points = PointSet(coordinates=coords)

# Option 1: Use AnomSmith if available
if is_anomsmith_available():
    result = detect_spatial_anomalies(points, method='isolation_forest')
    # Returns SpatialAnomalyResult compatible with AnomSmith
else:
    # Fallback: Use GeoSmith's built-in detection
    scores = AnomalyScores.from_points(points)
    result = SpatialAnomalyResult(scores=scores)

# Option 2: Convert to AnomSmith format (if available)
if is_anomsmith_available():
    as_result = result.to_anomsmith()  # Returns AnomSmith format
    # Use AnomSmith's advanced features
```

### Installation

```bash
# Install GeoSmith with AnomSmith support
pip install geosmith[anomsmith]

# Or install separately
pip install geosmith anomsmith
```

### Compatible Objects

GeoSmith provides objects that match AnomSmith's structure:

1. **AnomalyScores**: Compatible with AnomSmith's scoring interface
   - `to_anomsmith()`: Convert to AnomSmith format (if available)
   - `from_anomsmith()`: Create from AnomSmith object

2. **SpatialAnomalyResult**: Compatible with AnomSmith's result interface
   - `to_anomsmith()`: Convert to AnomSmith format (if available)
   - Works seamlessly with AnomSmith's detection methods

### Features

- **No Hard Dependency**: GeoSmith works without AnomSmith
- **Same Structure**: Objects match AnomSmith's interfaces
- **Seamless Integration**: When AnomSmith is available, objects work together
- **Type Safety**: Runtime validation matches AnomSmith's requirements

### Current Integration

- ✅ `AnomalyScores` - Compatible with AnomSmith's scoring interface
- ✅ `SpatialAnomalyResult` - Compatible with AnomSmith's result interface
- ✅ `detect_spatial_anomalies()` - Uses AnomSmith when available
- ✅ `to_anomsmith()` / `from_anomsmith()` - Conversion helpers

## TimeSmith Integration

**Repository**: https://github.com/kylejones200/timesmith

### What TimeSmith Provides

TimeSmith provides time series ML models and data structures.

### How GeoSmith Uses TimeSmith

GeoSmith provides compatible time series objects (`SeriesLike`, `PanelLike`, `TableLike`) that work seamlessly with TimeSmith:

```python
from geosmith import PointSet, SeriesLike, PanelLike, TableLike

# Create GeoSmith objects
points = PointSet(coordinates=coords)

# Convert to TimeSmith format (if available)
if timesmith_available():
    ts_series = points.to_timesmith()  # Returns TimeSmith SeriesLike
    # Use with TimeSmith models
```

### Installation

```bash
# Install GeoSmith with TimeSmith support
pip install geosmith[timesmith]

# Or install separately
pip install geosmith timesmith
```

### Compatible Objects

- **SeriesLike**: Compatible with TimeSmith's SeriesLike
- **PanelLike**: Compatible with TimeSmith's PanelLike
- **TableLike**: Compatible with TimeSmith's TableLike

## Usage Patterns

### Pattern 1: Optional Enhancement

```python
# Works with or without PlotSmith/AnomSmith
from geosmith import PointSet
from geosmith.workflows.plot import plot_points

points = PointSet(coordinates=coords)
ax = plot_points(points)  # Uses best available backend
```

### Pattern 2: Explicit Integration

```python
# Check availability and use explicitly
from geosmith.objects._anomsmith_compat import is_anomsmith_available

if is_anomsmith_available():
    from anomsmith import detect_anomalies
    result = detect_anomalies(data)
else:
    # Fallback implementation
    result = basic_anomaly_detection(data)
```

### Pattern 3: Conversion Between Libraries

```python
# Convert between *Smith libraries
from geosmith import PointSet
from geosmith.objects._anomsmith_compat import to_anomsmith_scores

points = PointSet(coordinates=coords)
scores = AnomalyScores.from_points(points)

# Use with AnomSmith (if available)
as_scores = scores.to_anomsmith()
```

## Benefits

1. **No Hard Dependencies**: GeoSmith works independently
2. **Optional Enhancements**: Install only what you need
3. **Seamless Integration**: When libraries are available, they work together
4. **Same Object Structure**: No conversion overhead
5. **Graceful Degradation**: Falls back to basic functionality when libraries aren't available

## Installation Options

```bash
# Minimal installation (core only)
pip install geosmith

# With specific integrations
pip install geosmith[plotsmith]      # PlotSmith support
pip install geosmith[anomsmith]      # AnomSmith support
pip install geosmith[timesmith]      # TimeSmith support

# All integrations
pip install geosmith[plotsmith,anomsmith,timesmith]

# Everything
pip install geosmith[all]
```

## Examples

See the examples directory for complete integration examples:

- `examples/anomsmith_integration.py` - AnomSmith integration
- `examples/timesmith_compatibility.py` - TimeSmith compatibility
- `examples/notebooks/` - Jupyter notebooks with PlotSmith visualization

## Summary

GeoSmith is designed to be **friendly but not dependent** on other *Smith libraries:

- ✅ Works independently without any *Smith libraries
- ✅ Automatically enhances when libraries are available
- ✅ Provides compatible object structures for seamless integration
- ✅ Graceful fallbacks when libraries aren't installed
- ✅ No conversion overhead when libraries are available

This design allows users to:
- Start with just GeoSmith
- Add PlotSmith for better plots when needed
- Add AnomSmith for anomaly detection when needed
- Add TimeSmith for time series analysis when needed
- Use all together for comprehensive geospatial ML workflows

