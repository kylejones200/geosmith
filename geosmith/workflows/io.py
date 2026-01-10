"""I/O helpers for loading and saving geospatial data."""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from geosmith.objects.geoindex import GeoIndex
from geosmith.objects.geotable import GeoTable
from geosmith.objects.pointset import PointSet
from geosmith.objects.polygonset import PolygonSet
from geosmith.objects.rastergrid import RasterGrid

logger = logging.getLogger(__name__)


def read_vector(
    path: Union[str, Path], crs: Optional[str] = None
) -> Union[GeoTable, PointSet, PolygonSet]:
    """Read vector data from file.

    Args:
        path: Path to vector file (shapefile, GeoJSON, etc.).
        crs: Optional CRS to assign.

    Returns:
        GeoTable, PointSet, or PolygonSet.
    """
    logger.info(f"Reading vector data from {path}")

    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError(
            "read_vector requires geopandas. Install with: pip install geosmith[vector]"
        )

    gdf = gpd.read_file(str(path))

    # Determine geometry type
    geom_type = gdf.geometry.iloc[0].geom_type

    # Convert to Layer 1 objects
    if geom_type == "Point":
        coords = np.array([(g.x, g.y) for g in gdf.geometry])
        index = GeoIndex(
            crs=crs or str(gdf.crs) if gdf.crs else None,
            bounds=(gdf.total_bounds[0], gdf.total_bounds[1], gdf.total_bounds[2], gdf.total_bounds[3]),
        )
        return PointSet(coordinates=coords, index=index)
    elif geom_type in ("Polygon", "MultiPolygon"):
        rings = []
        for geom in gdf.geometry:
            if geom.geom_type == "Polygon":
                exterior = np.array(geom.exterior.coords)
                holes = [np.array(ring.coords) for ring in geom.interiors]
                rings.append([exterior] + holes)
            elif geom.geom_type == "MultiPolygon":
                # Take first polygon
                first = geom.geoms[0]
                exterior = np.array(first.exterior.coords)
                rings.append([exterior])
        index = GeoIndex(
            crs=crs or str(gdf.crs) if gdf.crs else None,
            bounds=(gdf.total_bounds[0], gdf.total_bounds[1], gdf.total_bounds[2], gdf.total_bounds[3]),
        )
        return PolygonSet(rings=rings, index=index)
    else:
        # Create GeoTable for mixed/complex geometries
        # Convert geometries to objects
        geom_objects = []
        for geom in gdf.geometry:
            if geom.geom_type == "Point":
                coords = np.array([[geom.x, geom.y]])
                geom_objects.append(PointSet(coordinates=coords))
            elif geom.geom_type == "Polygon":
                exterior = np.array(geom.exterior.coords)
                geom_objects.append(PolygonSet(rings=[[exterior]]))
            else:
                # Fallback
                geom_objects.append(PointSet(coordinates=np.array([[0, 0]])))

        # Create DataFrame with geometry column
        data = gdf.drop(columns=["geometry"]).copy()
        data["geometry"] = geom_objects
        index = GeoIndex(
            crs=crs or str(gdf.crs) if gdf.crs else None,
            bounds=(gdf.total_bounds[0], gdf.total_bounds[1], gdf.total_bounds[2], gdf.total_bounds[3]),
        )
        return GeoTable(data=data, geometry_column="geometry", index=index)


def write_vector(
    data: Union[GeoTable, PointSet, PolygonSet],
    path: Union[str, Path],
    **kwargs,
) -> None:
    """Write vector data to file.

    Args:
        data: GeoTable, PointSet, or PolygonSet to write.
        path: Output file path.
        **kwargs: Additional arguments passed to geopandas.
    """
    logger.info(f"Writing vector data to {path}")

    try:
        import geopandas as gpd
        from shapely.geometry import Point, Polygon
    except ImportError:
        raise ImportError(
            "write_vector requires geopandas and shapely. "
            "Install with: pip install geosmith[vector]"
        )

    # Convert to geopandas
    if isinstance(data, PointSet):
        geoms = [Point(coord) for coord in data.coordinates]
        gdf = gpd.GeoDataFrame(geometry=geoms, crs=data.index.crs if data.index else None)
    elif isinstance(data, PolygonSet):
        geoms = []
        for polygon_rings in data.rings:
            if not polygon_rings:
                continue
            exterior = polygon_rings[0]
            holes = polygon_rings[1:] if len(polygon_rings) > 1 else None
            geoms.append(Polygon(exterior, holes=holes))
        gdf = gpd.GeoDataFrame(geometry=geoms, crs=data.index.crs if data.index else None)
    elif isinstance(data, GeoTable):
        # Convert geometry column
        geoms = []
        for geom_obj in data.data[data.geometry_column]:
            if isinstance(geom_obj, PointSet):
                geoms.append(Point(geom_obj.coordinates[0]))
            elif isinstance(geom_obj, PolygonSet):
                if geom_obj.rings and geom_obj.rings[0]:
                    exterior = geom_obj.rings[0][0]
                    geoms.append(Polygon(exterior))
                else:
                    geoms.append(None)
            else:
                geoms.append(None)
        gdf = gpd.GeoDataFrame(
            data.data.drop(columns=[data.geometry_column]),
            geometry=geoms,
            crs=data.index.crs if data.index else None,
        )
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

    gdf.to_file(str(path), **kwargs)


def read_raster(
    path: Union[str, Path], band: Optional[int] = None
) -> RasterGrid:
    """Read raster data from file.

    Args:
        path: Path to raster file (GeoTIFF, etc.).
        band: Optional band index to read (0-based).

    Returns:
        RasterGrid.
    """
    logger.info(f"Reading raster data from {path}")

    try:
        import rasterio
    except ImportError:
        raise ImportError(
            "read_raster requires rasterio. Install with: pip install geosmith[raster]"
        )

    with rasterio.open(str(path)) as src:
        if band is not None:
            data = src.read(band + 1)  # rasterio uses 1-based indexing
        else:
            data = src.read()

        transform = src.transform
        nodata = src.nodata
        crs = src.crs

        # Convert transform to tuple
        transform_tuple = (
            transform.a,
            transform.b,
            transform.c,
            transform.d,
            transform.e,
            transform.f,
        )

        # Get band names if available
        band_names = None
        if hasattr(src, "descriptions") and src.descriptions:
            band_names = [desc or f"band_{i}" for i, desc in enumerate(src.descriptions)]

        index = GeoIndex(
            crs=str(crs) if crs else None,
            bounds=src.bounds,
        )

        return RasterGrid(
            data=data,
            transform=transform_tuple,
            nodata=nodata,
            band_names=band_names,
            index=index,
        )


def write_raster(
    raster: RasterGrid,
    path: Union[str, Path],
    **kwargs,
) -> None:
    """Write raster data to file.

    Args:
        raster: RasterGrid to write.
        path: Output file path.
        **kwargs: Additional arguments passed to rasterio.
    """
    logger.info(f"Writing raster data to {path}")

    try:
        import rasterio
        from rasterio.transform import from_bounds
    except ImportError:
        raise ImportError(
            "write_raster requires rasterio. Install with: pip install geosmith[raster]"
        )

    data = raster.data
    if data.ndim == 2:
        data = data[np.newaxis, :, :]

    n_bands, height, width = data.shape

    # Reconstruct transform
    a, b, c, d, e, f = raster.transform
    transform = rasterio.Affine(a, b, c, d, e, f)

    # Determine CRS
    crs = None
    if raster.index and raster.index.crs:
        try:
            import pyproj
            crs = pyproj.CRS.from_string(raster.index.crs)
        except Exception:
            crs = raster.index.crs

    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=n_bands,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=raster.nodata,
        **kwargs,
    ) as dst:
        dst.write(data)



def load_csv_from_string(
    dash_upload_contents: str, encoding: Optional[str] = None
) -> pd.DataFrame:
    """Parse a Dash dcc.Upload contents string (data URL) assumed to be CSV.

    Args:
        dash_upload_contents: Base64-encoded CSV data URL string.
        encoding: Optional text encoding (e.g., 'utf-8', 'latin-1').

    Returns:
        DataFrame with parsed CSV data.

    Example:
        >>> from geosmith.workflows.io import load_csv_from_string
        >>>
        >>> # Dash upload content (data URL)
        >>> csv_data = "data:text/csv;base64,SGVhZGVyLFZhbHVlCjEsMi4zCjIsNC41"
        >>> df = load_csv_from_string(csv_data)
    """
    import base64
    import io

    if not dash_upload_contents:
        raise ValueError("No contents provided")

    try:
        header, b64data = dash_upload_contents.split(",", 1)
    except ValueError:
        raise ValueError(
            "Invalid contents format; expected data URL with base64 payload"
        )

    raw = base64.b64decode(b64data)
    buffer = io.BytesIO(raw)

    if encoding:
        return pd.read_csv(buffer, encoding=encoding)
    # pandas will infer encoding
    return pd.read_csv(buffer)

