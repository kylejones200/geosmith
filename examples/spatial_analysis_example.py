"""Example: Spatial autocorrelation and hotspot detection.

Demonstrates Moran's I, Geary's C, and Getis-Ord Gi* hotspot detection
using GeoSmith's spatial analysis tools.
"""

import numpy as np

from geosmith import PointSet
from geosmith.primitives.spatial_analysis import (
    create_knn_weights,
    gearys_c,
    getis_ord_gi_star,
    morans_i,
)


def main():
    """Run spatial analysis example."""
    print("=" * 60)
    print("Spatial Autocorrelation and Hotspot Detection Example")
    print("=" * 60)

    # Create synthetic data with spatial clustering
    print("\n1. Creating synthetic spatially correlated data...")
    np.random.seed(42)
    n_points = 100

    # Create coordinates
    x = np.random.rand(n_points) * 1000
    y = np.random.rand(n_points) * 1000
    coords = np.column_stack([x, y])
    points = PointSet(coordinates=coords)

    # Create values with spatial clustering (high values cluster together)
    # Use distance-based clustering
    cluster_centers = np.array([[200, 200], [800, 800]])
    values = np.zeros(n_points)

    for i in range(n_points):
        dist_to_cluster1 = np.sqrt(
            (coords[i, 0] - cluster_centers[0, 0]) ** 2
            + (coords[i, 1] - cluster_centers[0, 1]) ** 2
        )
        dist_to_cluster2 = np.sqrt(
            (coords[i, 0] - cluster_centers[1, 0]) ** 2
            + (coords[i, 1] - cluster_centers[1, 1]) ** 2
        )
        # Higher values near cluster centers
        values[i] = (
            10 * np.exp(-dist_to_cluster1 / 100)
            + 10 * np.exp(-dist_to_cluster2 / 100)
            + np.random.randn() * 2
        )

    print(f"Created {n_points} points with spatially clustered values")
    print(f"Value statistics: mean={values.mean():.2f}, std={values.std():.2f}")

    # Create spatial weights (K-nearest neighbors)
    print("\n2. Creating spatial weights matrix...")
    weights = create_knn_weights(points, k=8)
    print(f"Spatial weights: {weights}")

    # Compute Moran's I
    print("\n3. Computing Moran's I (spatial autocorrelation)...")
    moran_result = morans_i(values, weights)
    print(f"Moran's I: {moran_result.I:.4f}")
    print(f"Expected I: {moran_result.expected_I:.4f}")
    print(f"Z-score: {moran_result.z_score:.2f}")
    print(f"P-value: {moran_result.p_value:.4f}")

    if moran_result.p_value < 0.05:
        if moran_result.I > 0:
            print("  → Positive spatial autocorrelation detected (clustering)")
        else:
            print("  → Negative spatial autocorrelation detected (dispersion)")
    else:
        print("  → No significant spatial autocorrelation")

    # Compute Geary's C
    print("\n4. Computing Geary's C (alternative autocorrelation measure)...")
    geary_result = gearys_c(values, weights)
    print(f"Geary's C: {geary_result.C:.4f}")
    print(f"Expected C: {geary_result.expected_C:.4f}")
    print(f"Z-score: {geary_result.z_score:.2f}")
    print(f"P-value: {geary_result.p_value:.4f}")

    if geary_result.p_value < 0.05:
        if geary_result.C < 1:
            print("  → Positive spatial autocorrelation (similar values cluster)")
        else:
            print("  → Negative spatial autocorrelation (dissimilar values cluster)")
    else:
        print("  → No significant spatial autocorrelation")

    # Hotspot detection using Getis-Ord Gi*
    print("\n5. Detecting hotspots and coldspots (Getis-Ord Gi*)...")
    hotspot_result = getis_ord_gi_star(values, weights)
    print(f"Hotspot detection: {hotspot_result}")
    print(f"  Significant hotspots: {hotspot_result.hotspots.sum()}")
    print(f"  Significant coldspots: {hotspot_result.coldspots.sum()}")

    # Display hotspot locations
    if hotspot_result.hotspots.sum() > 0:
        print("\n  Hotspot locations (high values):")
        hotspot_coords = coords[hotspot_result.hotspots]
        for i, (x, y) in enumerate(hotspot_coords[:5]):  # Show first 5
            idx = np.where(hotspot_result.hotspots)[0][i]
            print(f"    Point {idx}: ({x:.1f}, {y:.1f}), value={values[idx]:.2f}, "
                  f"z={hotspot_result.z_scores[idx]:.2f}")

    if hotspot_result.coldspots.sum() > 0:
        print("\n  Coldspot locations (low values):")
        coldspot_coords = coords[hotspot_result.coldspots]
        for i, (x, y) in enumerate(coldspot_coords[:5]):  # Show first 5
            idx = np.where(hotspot_result.coldspots)[0][i]
            print(f"    Point {idx}: ({x:.1f}, {y:.1f}), value={values[idx]:.2f}, "
                  f"z={hotspot_result.z_scores[idx]:.2f}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Spatial autocorrelation: {'Significant' if moran_result.p_value < 0.05 else 'Not significant'}")
    print(f"  Moran's I: {moran_result.I:.4f} ({'clustering' if moran_result.I > 0 else 'dispersion'})")
    print(f"  Hotspots detected: {hotspot_result.hotspots.sum()}")
    print(f"  Coldspots detected: {hotspot_result.coldspots.sum()}")
    print("=" * 60)


if __name__ == "__main__":
    main()

