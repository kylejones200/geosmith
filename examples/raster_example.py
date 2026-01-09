"""Example: Raster processing pipeline (clip + resample + zonal stats).

This example demonstrates the raster processing pipeline:
1. Clip raster to polygon boundary
2. Resample raster to new resolution
3. Compute zonal statistics
"""

import logging
import numpy as np
from pathlib import Path

from geosmith import (
    RasterGrid,
    PolygonSet,
    GeoIndex,
    process_raster,
    zonal_stats,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
np.random.seed(42)


def create_synthetic_raster(shape=(100, 100), n_bands=1) -> RasterGrid:
    """Create synthetic raster data."""
    if n_bands == 1:
        data = np.random.rand(*shape) * 100
    else:
        data = np.random.rand(n_bands, *shape) * 100

    transform = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)  # 1 unit per pixel
    index = GeoIndex(
        crs="EPSG:4326",
        bounds=(0, 0, shape[1], shape[0]),
    )
    return RasterGrid(data=data, transform=transform, index=index)


def create_synthetic_polygons(n: int = 3) -> PolygonSet:
    """Create synthetic polygon zones."""
    rings = []
    for i in range(n):
        center_x = 20 + i * 30
        center_y = 20 + i * 30
        size = 10
        # Create square polygon
        ring = np.array([
            [center_x - size, center_y - size],
            [center_x + size, center_y - size],
            [center_x + size, center_y + size],
            [center_x - size, center_y + size],
            [center_x - size, center_y - size],  # Close ring
        ])
        rings.append([ring])

    index = GeoIndex(
        crs="EPSG:4326",
        bounds=(0, 0, 100, 100),
    )
    return PolygonSet(rings=rings, index=index)


def main():
    """Run raster processing pipeline example."""
    logger.info("Starting raster processing pipeline example")

    # Create synthetic data
    logger.info("Creating synthetic raster data")
    raster = create_synthetic_raster(shape=(100, 100), n_bands=1)
    logger.info(f"Created raster with shape {raster.data.shape}")

    logger.info("Creating synthetic polygon zones")
    polygons = create_synthetic_polygons(n=3)
    logger.info(f"Created {len(polygons.rings)} polygon zones")

    # Step 1: Clip raster
    logger.info("Step 1: Clipping raster to polygon boundary")
    clipped = process_raster(
        raster,
        operations={
            "clip": {"geometry": polygons},
        },
    )
    logger.info(f"Clipped raster shape: {clipped.data.shape}")

    # Step 2: Resample raster
    logger.info("Step 2: Resampling raster to new resolution")
    target_transform = (2.0, 0.0, 0.0, 0.0, -2.0, 0.0)  # 2 units per pixel
    resampled = process_raster(
        raster,
        operations={
            "resample": {
                "target_transform": target_transform,
                "target_shape": (50, 50),
                "method": "nearest",
            },
        },
    )
    logger.info(f"Resampled raster shape: {resampled.data.shape}")

    # Step 3: Zonal statistics
    logger.info("Step 3: Computing zonal statistics")
    stats = zonal_stats(
        raster,
        polygons,
        reducer="mean",
    )
    logger.info(f"Computed zonal statistics for {len(stats)} zones")
    logger.info(f"Statistics:\n{stats}")

    # Save results
    output_dir = Path(__file__).parent / "out"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save zonal statistics
    stats.to_csv(output_dir / "zonal_stats.csv", index=False)
    logger.info(f"Saved zonal statistics to {output_dir / 'zonal_stats.csv'}")

    logger.info("Raster processing pipeline example completed successfully")


if __name__ == "__main__":
    main()

