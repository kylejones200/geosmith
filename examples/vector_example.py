"""Example: Vector feature pipeline (buffer + spatial join + distance to nearest).

This example demonstrates the vector feature pipeline:
1. Create buffer around points
2. Perform spatial join
3. Compute distance to nearest neighbors
"""

import logging
import numpy as np
from pathlib import Path

from geosmith import (
    PointSet,
    PolygonSet,
    GeoIndex,
    make_features,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
np.random.seed(42)


def create_synthetic_points(n: int = 10) -> PointSet:
    """Create synthetic point data."""
    coords = np.random.rand(n, 2) * 100  # Points in 0-100 range
    index = GeoIndex(
        crs="EPSG:4326",
        bounds=(0, 0, 100, 100),
    )
    return PointSet(coordinates=coords, index=index)


def create_synthetic_polygons(n: int = 5) -> PolygonSet:
    """Create synthetic polygon data."""
    rings = []
    for i in range(n):
        center_x = np.random.rand() * 100
        center_y = np.random.rand() * 100
        size = 5 + np.random.rand() * 10
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
    """Run vector feature pipeline example."""
    logger.info("Starting vector feature pipeline example")

    # Create synthetic data
    logger.info("Creating synthetic point data")
    points = create_synthetic_points(n=20)
    logger.info(f"Created {len(points.coordinates)} points")

    logger.info("Creating synthetic polygon data")
    polygons = create_synthetic_polygons(n=5)
    logger.info(f"Created {len(polygons.rings)} polygons")

    # Step 1: Buffer points
    logger.info("Step 1: Buffering points")
    buffered = make_features(
        points,
        operations={
            "buffer": {"distance": 5.0},
        },
    )
    logger.info(f"Created {len(buffered.rings)} buffered polygons")

    # Step 2: Spatial join (points in polygons)
    logger.info("Step 2: Performing spatial join")
    join_result = make_features(
        points,
        operations={
            "spatial_join": {
                "right": polygons,
                "how": "inner",
                "predicate": "intersects",
            },
        },
    )
    logger.info(f"Spatial join found {len(join_result)} matches")

    # Step 3: Distance to nearest
    logger.info("Step 3: Computing distance to nearest neighbors")
    query_points = create_synthetic_points(n=5)
    distances = make_features(
        query_points,
        operations={
            "distance_to_nearest": {
                "target": points,
                "k": 3,
            },
        },
    )
    logger.info(f"Computed distances for {len(distances)} query points")
    logger.info(f"Distance columns: {list(distances.columns)}")

    # Save results
    output_dir = Path(__file__).parent / "out"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save join results
    join_result.to_csv(output_dir / "spatial_join_results.csv", index=False)
    logger.info(f"Saved spatial join results to {output_dir / 'spatial_join_results.csv'}")

    # Save distance results
    distances.to_csv(output_dir / "distance_results.csv", index=False)
    logger.info(f"Saved distance results to {output_dir / 'distance_results.csv'}")

    logger.info("Vector feature pipeline example completed successfully")


if __name__ == "__main__":
    main()

