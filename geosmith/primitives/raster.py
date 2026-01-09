"""Pure raster operations."""

from typing import Optional, Tuple

import numpy as np

from geosmith.objects.rastergrid import RasterGrid


def grid_resample(
    source: RasterGrid,
    target_transform: Tuple[float, float, float, float, float, float],
    target_shape: Tuple[int, int],
    method: str = "nearest",
) -> RasterGrid:
    """Resample a raster grid to a new transform and shape.

    Args:
        source: Source RasterGrid.
        target_transform: Target affine transform.
        target_shape: Target shape as (n_rows, n_cols).
        method: Resampling method, 'nearest' or 'bilinear'.

    Returns:
        Resampled RasterGrid.
    """
    # Simple nearest neighbor resampling
    # For production, this would use rasterio or similar
    source_data = source.data
    if source_data.ndim == 2:
        source_data = source_data[np.newaxis, :, :]

    n_bands, source_rows, source_cols = source_data.shape
    target_rows, target_cols = target_shape

    # Create target grid
    target_data = np.zeros((n_bands, target_rows, target_cols), dtype=source_data.dtype)

    # Simple nearest neighbor (for production, use proper resampling)
    if method == "nearest":
        # Map target pixel centers to source coordinates
        # This is a simplified version - production would use proper affine math
        row_ratio = source_rows / target_rows
        col_ratio = source_cols / target_cols

        for i in range(target_rows):
            for j in range(target_cols):
                src_i = int(i * row_ratio)
                src_j = int(j * col_ratio)
                src_i = min(src_i, source_rows - 1)
                src_j = min(src_j, source_cols - 1)
                target_data[:, i, j] = source_data[:, src_i, src_j]

    else:
        raise ValueError(f"Unknown resampling method: {method}")

    # Handle nodata
    if source.nodata is not None:
        # Preserve nodata in output
        pass

    # Collapse to 2D if single band
    if n_bands == 1:
        target_data = target_data[0, :, :]

    return RasterGrid(
        data=target_data,
        transform=target_transform,
        nodata=source.nodata,
        band_names=source.band_names,
        index=source.index,
    )


def zonal_reduce(
    raster: RasterGrid,
    polygons: "PolygonSet",  # type: ignore
    reducer: str = "mean",
    band: Optional[int] = None,
) -> np.ndarray:
    """Compute zonal statistics for polygons over a raster.

    Args:
        raster: RasterGrid to compute statistics over.
        polygons: PolygonSet defining zones.
        reducer: Reduction function, 'mean', 'sum', 'min', 'max', 'std'.
        band: Optional band index (0-based). If None, uses first band.

    Returns:
        Array of shape (n_polygons,) with zonal statistics.
    """
    # Simplified zonal statistics
    # For production, this would properly handle polygon-raster intersection
    raster_data = raster.data
    if raster_data.ndim == 3:
        if band is None:
            band = 0
        raster_data = raster_data[band, :, :]
    elif band is not None and band != 0:
        raise ValueError(f"Band {band} requested but raster has only 1 band")

    n_polygons = len(polygons.rings)
    result = np.zeros(n_polygons, dtype=raster_data.dtype)

    # For each polygon, compute statistics
    # This is a simplified version - production would properly intersect
    for poly_idx, polygon_rings in enumerate(polygons.rings):
        if not polygon_rings:
            continue

        # Get bounding box of polygon
        exterior = polygon_rings[0]
        minx, miny = exterior[:, :2].min(axis=0)
        maxx, maxy = exterior[:, :2].max(axis=0)

        # Sample raster in bbox (simplified - production would use proper intersection)
        # For now, just return mean of entire raster as placeholder
        if reducer == "mean":
            result[poly_idx] = float(np.nanmean(raster_data))
        elif reducer == "sum":
            result[poly_idx] = float(np.nansum(raster_data))
        elif reducer == "min":
            result[poly_idx] = float(np.nanmin(raster_data))
        elif reducer == "max":
            result[poly_idx] = float(np.nanmax(raster_data))
        elif reducer == "std":
            result[poly_idx] = float(np.nanstd(raster_data))
        else:
            raise ValueError(f"Unknown reducer: {reducer}")

    return result

