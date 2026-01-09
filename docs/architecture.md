GeoSmith follows a four layer architecture.

The first layer holds objects. Objects represent spatial structure. Points, lines, polygons, rasters, and coordinate metadata live here. Objects do not depend on GIS libraries.

The second layer holds primitives. Primitives implement geometry and raster operations. Distance, intersection, resampling, and aggregation live here. Primitives never load files and never plot.

The third layer holds tasks. Tasks bind intent. A task defines what spatial operation is required. Feature extraction, raster processing, and change analysis live here. Tasks validate inputs and orchestrate primitives.

The fourth layer holds workflows. Workflows handle reality. They read and write files, integrate with geopandas or rasterio when available, and produce outputs ready for downstream systems.

Each layer imports only from layers below it. This separation keeps spatial logic testable and reusable.

GeoSmith stands alone but integrates cleanly with the rest of the smith ecosystem. When time enters the problem, GeoSmith uses shared typing from Timesmith rather than inventing new conventions.

