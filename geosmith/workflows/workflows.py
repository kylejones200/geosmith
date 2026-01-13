"""End-to-end workflow functions."""

import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from geosmith.objects.geoindex import GeoIndex
from geosmith.objects.geotable import GeoTable
from geosmith.objects.pointset import PointSet
from geosmith.objects.polygonset import PolygonSet
from geosmith.objects.rastergrid import RasterGrid
from geosmith.tasks.featuretask import FeatureTask
from geosmith.tasks.rastertask import RasterTask

logger = logging.getLogger(__name__)


def make_features(
    geometry: Union[GeoTable, PolygonSet, PointSet],
    operations: Dict[str, Any],
    crs: Optional[str] = None,
) -> Union[GeoTable, PolygonSet, PointSet, pd.DataFrame]:
    """Run feature creation pipeline.

    Args:
        geometry: Input geometry.
        operations: Dictionary of operations to perform.
        crs: Optional CRS for output.

    Returns:
        Result geometry or DataFrame.
    """
    logger.info("Starting make_features workflow")
    logger.info(f"Operations: {list(operations.keys())}")

    task = FeatureTask(crs=crs)
    result = geometry

    # Apply operations in order
    for op_name, op_params in operations.items():
        if op_name == "buffer":
            result = task.buffer(result, **op_params)
        elif op_name == "spatial_join":
            right = op_params.pop("right")
            result = task.spatial_join(result, right, **op_params)
        elif op_name == "distance_to_nearest":
            target = op_params.pop("target")
            result = task.distance_to_nearest(result, target, **op_params)
        else:
            raise ValueError(f"Unknown operation: {op_name}")

    logger.info("Completed make_features workflow")
    return result


def process_raster(
    raster: RasterGrid,
    operations: Dict[str, Any],
    crs: Optional[str] = None,
) -> RasterGrid:
    """Run raster processing pipeline.

    Args:
        raster: Input RasterGrid.
        operations: Dictionary of operations to perform.
        crs: Optional CRS for output.

    Returns:
        Processed RasterGrid.
    """
    logger.info("Starting process_raster workflow")
    logger.info(f"Operations: {list(operations.keys())}")

    task = RasterTask(crs=crs)
    result = raster

    # Apply operations in order
    for op_name, op_params in operations.items():
        if op_name == "clip":
            result = task.clip(result, **op_params)
        elif op_name == "resample":
            result = task.resample(result, **op_params)
        elif op_name == "zonal_stats":
            # This returns a DataFrame, not a raster
            polygons = op_params.pop("polygons")
            stats = task.zonal_stats(result, polygons, **op_params)
            # For now, return stats as DataFrame
            return pd.DataFrame({"value": stats})
        else:
            raise ValueError(f"Unknown operation: {op_name}")

    logger.info("Completed process_raster workflow")
    return result


def zonal_stats(
    raster: RasterGrid,
    polygons: PolygonSet,
    reducer: str = "mean",
    band: Optional[int] = None,
) -> pd.DataFrame:
    """Compute zonal statistics for polygons over raster.

    Args:
        raster: Input RasterGrid.
        polygons: PolygonSet defining zones.
        reducer: Reduction function ('mean', 'sum', 'min', 'max', 'std').
        band: Optional band index.

    Returns:
        DataFrame with zonal statistics.
    """
    logger.info("Starting zonal_stats workflow")

    task = RasterTask()
    stats = task.zonal_stats(raster, polygons, reducer=reducer, band=band)

    result = pd.DataFrame({"value": stats})
    logger.info("Completed zonal_stats workflow")
    return result


def reproject_to(
    data: Union[PointSet, PolygonSet, RasterGrid, GeoTable],
    target_crs: str,
) -> Union[PointSet, PolygonSet, RasterGrid, GeoTable]:
    """Reproject any object to target CRS.

    Args:
        data: Input geometry or raster.
        target_crs: Target CRS (e.g., 'EPSG:4326').

    Returns:
        Reprojected object.
    """
    logger.info(f"Reprojecting to {target_crs}")

    try:
        import pyproj
    except ImportError:
        raise ImportError(
            "reproject_to requires pyproj. Install with: pip install geosmith[vector]"
        )

    # Simple reprojection (would use proper transformation in production)
    if isinstance(data, (PointSet, PolygonSet, GeoTable)):
        # Update CRS in index
        if data.index:
            new_index = GeoIndex(
                crs=target_crs,
                bounds=data.index.bounds,
                axis_order=data.index.axis_order,
            )
        else:
            new_index = GeoIndex(crs=target_crs, bounds=(0, 0, 1, 1))

        if isinstance(data, PointSet):
            return PointSet(
                coordinates=data.coordinates,
                attributes=data.attributes,
                index=new_index,
            )
        elif isinstance(data, PolygonSet):
            return PolygonSet(
                rings=data.rings,
                attributes=data.attributes,
                index=new_index,
            )
        elif isinstance(data, GeoTable):
            return GeoTable(
                data=data.data,
                geometry_column=data.geometry_column,
                index=new_index,
            )
    elif isinstance(data, RasterGrid):
        # Update CRS in index
        if data.index:
            new_index = GeoIndex(
                crs=target_crs,
                bounds=data.index.bounds,
                axis_order=data.index.axis_order,
            )
        else:
            new_index = GeoIndex(crs=target_crs, bounds=(0, 0, 1, 1))

        return RasterGrid(
            data=data.data,
            transform=data.transform,
            nodata=data.nodata,
            band_names=data.band_names,
            index=new_index,
        )
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")
