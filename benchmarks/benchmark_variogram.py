"""Performance benchmarks for variogram computation."""

import time
from typing import Dict

import numpy as np

from geosmith import PointSet
from geosmith.primitives.variogram import (
    compute_experimental_cross_variogram,
    compute_experimental_variogram,
    fit_variogram_model,
)


def benchmark_variogram_computation(
    n_samples: int = 1000,
    n_lags: int = 15,
) -> Dict[str, float]:
    """Benchmark experimental variogram computation.

    Args:
        n_samples: Number of sample points.
        n_lags: Number of lag bins.

    Returns:
        Dictionary with timing results.
    """
    np.random.seed(42)

    # Create sample data
    coords = np.random.rand(n_samples, 2) * 1000
    values = np.random.rand(n_samples) * 10
    points = PointSet(coordinates=coords)

    # Benchmark variogram computation
    start = time.perf_counter()
    lags, semi_vars, n_pairs = compute_experimental_variogram(
        points, values, n_lags=n_lags
    )
    compute_time = time.perf_counter() - start

    # Benchmark model fitting
    start = time.perf_counter()
    variogram_model = fit_variogram_model(lags, semi_vars, model_type="spherical")
    fit_time = time.perf_counter() - start

    return {
        "n_samples": n_samples,
        "n_lags": n_lags,
        "compute_time_seconds": compute_time,
        "fit_time_seconds": fit_time,
        "total_time_seconds": compute_time + fit_time,
        "samples_per_second": n_samples / compute_time if compute_time > 0 else 0,
    }


def benchmark_cross_variogram(
    n_primary: int = 200,
    n_secondary: int = 300,
    n_lags: int = 15,
) -> Dict[str, float]:
    """Benchmark cross-variogram computation.

    Args:
        n_primary: Number of primary variable samples.
        n_secondary: Number of secondary variable samples.
        n_lags: Number of lag bins.

    Returns:
        Dictionary with timing results.
    """
    np.random.seed(42)

    primary_coords = np.random.rand(n_primary, 2) * 1000
    primary_values = np.random.rand(n_primary) * 10
    primary_points = PointSet(coordinates=primary_coords)

    secondary_coords = np.random.rand(n_secondary, 2) * 1000
    secondary_values = np.random.rand(n_secondary) * 10
    secondary_points = PointSet(coordinates=secondary_coords)

    # Benchmark cross-variogram computation
    start = time.perf_counter()
    lags, cross_semi_vars, n_pairs = compute_experimental_cross_variogram(
        primary_points,
        primary_values,
        secondary_points,
        secondary_values,
        n_lags=n_lags,
    )
    compute_time = time.perf_counter() - start

    return {
        "n_primary": n_primary,
        "n_secondary": n_secondary,
        "n_lags": n_lags,
        "compute_time_seconds": compute_time,
        "total_pairs": n_primary * n_secondary,
        "pairs_per_second": (n_primary * n_secondary) / compute_time
        if compute_time > 0
        else 0,
    }


def benchmark_variogram_scalability() -> Dict[str, Dict[str, float]]:
    """Benchmark variogram computation across different sample sizes.

    Returns:
        Dictionary with results for different sizes.
    """
    results = {}

    sizes = [
        ("small", 100),
        ("medium", 500),
        ("large", 1000),
        ("xlarge", 5000),
    ]

    for size_name, n_samples in sizes:
        print(f"  Benchmarking {size_name} ({n_samples} samples)...")
        results[size_name] = benchmark_variogram_computation(n_samples=n_samples)

    return results


def run_all_variogram_benchmarks() -> Dict[str, Dict]:
    """Run all variogram benchmarks and return results.

    Returns:
        Dictionary with all benchmark results.
    """
    results = {}

    print("Benchmarking variogram computation scalability...")
    results["variogram_scalability"] = benchmark_variogram_scalability()

    print("Benchmarking cross-variogram computation...")
    results["cross_variogram"] = benchmark_cross_variogram(200, 300, 15)

    return results


if __name__ == "__main__":
    """Run benchmarks and print results."""
    results = run_all_variogram_benchmarks()

    print("\n" + "=" * 60)
    print("VARIOGRAM PERFORMANCE BENCHMARKS")
    print("=" * 60)

    print("\nVariogram Computation Scalability:")
    for size, data in results["variogram_scalability"].items():
        print(f"  {size:8s}: {data['n_samples']:5d} samples")
        print(f"            Compute: {data['compute_time_seconds']*1000:6.2f} ms")
        print(f"            Fit: {data['fit_time_seconds']*1000:6.2f} ms")
        print(f"            Total: {data['total_time_seconds']*1000:6.2f} ms")
        print(f"            Throughput: {data['samples_per_second']:8.0f} samples/s")

    print("\nCross-Variogram Computation:")
    cv = results["cross_variogram"]
    print(f"  Primary samples: {cv['n_primary']}")
    print(f"  Secondary samples: {cv['n_secondary']}")
    print(f"  Compute time: {cv['compute_time_seconds']*1000:6.2f} ms")
    print(f"  Throughput: {cv['pairs_per_second']:8.0f} pairs/s")

