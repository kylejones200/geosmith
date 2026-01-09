# GeoSmith

*Smith for geospatial ML models with strict 4-layer architecture.

[![PyPI version](https://badge.fury.io/py/geosmith.svg)](https://badge.fury.io/py/geosmith)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://readthedocs.org/projects/geosmith/badge/?version=latest)](https://geosmith.readthedocs.io/)

## Features

- **4-Layer Architecture**: Clean separation between objects, primitives, tasks, and workflows
- **Optional Integrations**: Works seamlessly with PlotSmith, AnomSmith, and TimeSmith
- **Geostatistics**: Variogram analysis, kriging, IDW interpolation
- **Mining Tools**: Block models, drillhole processing, ore grade estimation
- **I/O Support**: GRDECL, LAS, vector/raster formats
- **No Hard Dependencies**: Core works with just NumPy and Pandas

## Installation

```bash
# Core installation
pip install geosmith

# With optional integrations
pip install geosmith[plotsmith,anomsmith,timesmith]

# Everything
pip install geosmith[all]
```

## Quick Start

```python
from geosmith import PointSet, OrdinaryKriging
from geosmith.primitives.variogram import fit_variogram_model

# Create sample points
points = PointSet(coordinates=sample_coords)

# Fit variogram and kriging
variogram = fit_variogram_model(lags, semi_vars)
kriging = OrdinaryKriging(variogram_model=variogram)
kriging.fit(points, values)

# Predict
result = kriging.predict(query_points)
```

## *Smith Family Integration

GeoSmith works seamlessly with other *Smith libraries:

- **PlotSmith**: Publication-ready plots (optional) - https://github.com/kylejones200/plotsmith
- **AnomSmith**: Spatial anomaly detection (optional) - https://github.com/kylejones200/anomsmith
- **TimeSmith**: Time series compatibility (optional) - https://github.com/kylejones200/timesmith

See [SMITH_FAMILY_INTEGRATION.md](SMITH_FAMILY_INTEGRATION.md) for details.

## Documentation

- [Architecture](ARCHITECTURE.md) - 4-layer architecture overview
- [Migration Guide](MIGRATION_GUIDE.md) - Migrating from GeoSuite
- [Smith Family Integration](SMITH_FAMILY_INTEGRATION.md) - PlotSmith, AnomSmith, TimeSmith integration
- [GeoSuite Migration Assessment](GEOSUITE_MIGRATION_ASSESSMENT.md) - What's been migrated from GeoSuite
