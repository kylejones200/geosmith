"""Example: Drillhole processing workflow.

Demonstrates processing drillhole collar and assay data,
computing 3D coordinates, and creating a PointSet.
"""

import numpy as np
import pandas as pd

from geosmith import PointSet
from geosmith.workflows.drillhole import (
    compute_3d_coordinates,
    merge_collar_assay,
    process_drillhole_data,
)


def main():
    """Run drillhole processing example."""
    print("=" * 60)
    print("Drillhole Processing Example")
    print("=" * 60)

    # Create synthetic collar data
    print("\n1. Creating synthetic collar data...")
    collar_data = {
        "HOLEID": ["DH001", "DH002", "DH003"],
        "EASTING": [100.0, 150.0, 200.0],
        "NORTHING": [500.0, 550.0, 600.0],
        "RL": [100.0, 105.0, 110.0],
    }
    collar_df = pd.DataFrame(collar_data)
    print(f"Collar data:\n{collar_df}")

    # Create synthetic assay data
    print("\n2. Creating synthetic assay data...")
    assay_data = {
        "HOLEID": ["DH001", "DH001", "DH002", "DH002", "DH003"],
        "FROM": [0.0, 10.0, 0.0, 10.0, 0.0],
        "TO": [10.0, 20.0, 10.0, 20.0, 10.0],
        "Au": [2.5, 1.8, 3.2, 2.1, 2.9],
    }
    assay_df = pd.DataFrame(assay_data)
    print(f"Assay data:\n{assay_df}")

    # Process drillhole data (auto-detect columns)
    print("\n3. Processing drillhole data (auto-detecting columns)...")
    column_map = process_drillhole_data(collar_df, assay_df)
    print(f"Detected columns: {column_map}")

    # Merge collar and assay data
    print("\n4. Merging collar and assay data...")
    merged_df = merge_collar_assay(collar_df, assay_df, column_map)
    print(f"Merged data:\n{merged_df}")

    # Compute 3D coordinates
    print("\n5. Computing 3D coordinates...")
    points = compute_3d_coordinates(merged_df, assume_vertical=True)
    print(f"PointSet created with {len(points.coordinates)} points")
    print(f"Coordinates shape: {points.coordinates.shape}")
    print(f"Attributes: {list(points.attributes.keys())}")

    # Display sample points
    print("\n6. Sample points:")
    for i in range(min(3, len(points.coordinates))):
        x, y, z = points.coordinates[i]
        grade = points.attributes["grade"][i]
        hole_id = points.attributes["hole_id"][i]
        print(f"  {hole_id}: ({x:.1f}, {y:.1f}, {z:.1f}) - Grade: {grade:.2f}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

