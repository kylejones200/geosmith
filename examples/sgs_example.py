"""Example: Sequential Gaussian Simulation (SGS) for uncertainty quantification.

Demonstrates using SGS to generate multiple realizations of a spatial field
and compute exceedance probabilities.
"""

import numpy as np

from geosmith import PointSet
from geosmith.primitives.simulation import (
    compute_exceedance_probability,
    compute_simulation_statistics,
    sequential_gaussian_simulation,
)
from geosmith.primitives.variogram import (
    compute_experimental_variogram,
    fit_variogram_model,
)


def main():
    """Run SGS example."""
    print("=" * 60)
    print("Sequential Gaussian Simulation (SGS) Example")
    print("=" * 60)

    # Create synthetic sample data with spatial correlation
    print("\n1. Creating synthetic sample data...")
    np.random.seed(42)
    n_samples = 50

    # Create spatially correlated samples
    x = np.random.rand(n_samples) * 1000
    y = np.random.rand(n_samples) * 1000
    # Create values with spatial correlation
    values = (
        x * 0.1
        + y * 0.15
        + np.sin(x / 100) * 10
        + np.random.randn(n_samples) * 5
    )

    samples = PointSet(coordinates=np.column_stack([x, y]))
    print(f"Created {len(samples.coordinates)} sample points")

    # Compute experimental variogram
    print("\n2. Computing experimental variogram...")
    lags, semi_vars, n_pairs = compute_experimental_variogram(
        samples, values, n_lags=15
    )

    # Fit variogram model
    print("\n3. Fitting variogram model...")
    variogram = fit_variogram_model(lags, semi_vars, model_type="spherical")
    print(f"Fitted {variogram.model_type} model:")
    print(f"  Nugget: {variogram.nugget:.4f}")
    print(f"  Sill: {variogram.sill:.4f}")
    print(f"  Range: {variogram.range_param:.4f}")

    # Create query points (grid)
    print("\n4. Creating query grid...")
    query_x = np.linspace(0, 1000, 20)
    query_y = np.linspace(0, 1000, 20)
    query_xx, query_yy = np.meshgrid(query_x, query_y)
    query_coords = np.column_stack([query_xx.ravel(), query_yy.ravel()])
    query_points = PointSet(coordinates=query_coords)
    print(f"Created {len(query_points.coordinates)} query points")

    # Run SGS
    print("\n5. Running Sequential Gaussian Simulation...")
    n_realizations = 50
    realizations = sequential_gaussian_simulation(
        samples,
        values,
        query_points,
        variogram,
        n_realizations=n_realizations,
        random_seed=42,
    )
    print(f"Generated {realizations.shape[0]} realizations")
    print(f"Each realization has {realizations.shape[1]} points")

    # Compute statistics
    print("\n6. Computing simulation statistics...")
    stats = compute_simulation_statistics(realizations)
    print(f"Mean: {stats['mean'].mean():.2f} ± {stats['std'].mean():.2f}")
    print(f"P10: {stats['p10'].mean():.2f}")
    print(f"P50 (median): {stats['p50'].mean():.2f}")
    print(f"P90: {stats['p90'].mean():.2f}")

    # Compute exceedance probability
    print("\n7. Computing exceedance probabilities...")
    threshold = np.percentile(values, 75)  # 75th percentile of samples
    prob_exceed = compute_exceedance_probability(realizations, threshold)
    print(f"Threshold: {threshold:.2f}")
    print(f"Mean exceedance probability: {prob_exceed.mean():.2%}")
    print(f"Points with >50% probability: {(prob_exceed > 0.5).sum()}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

