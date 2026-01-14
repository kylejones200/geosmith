"""Performance benchmarks for simulation operations."""

import time
from typing import Dict

import numpy as np

from geosmith import PointSet
from geosmith.primitives.simulation import sequential_gaussian_simulation
from geosmith.primitives.variogram import VariogramModel


def benchmark_sgs(
    n_samples: int = 50,
    n_targets: int = 1000,
    n_realizations: int = 10,
) -> Dict[str, float]:
    """Benchmark Sequential Gaussian Simulation performance.

    Args:
        n_samples: Number of sample points.
        n_targets: Number of target points to simulate.
        n_realizations: Number of realizations to generate.

    Returns:
        Dictionary with timing results.
    """
    np.random.seed(42)

    # Create sample data
    sample_coords = np.random.rand(n_samples, 2) * 1000
    sample_values = np.random.rand(n_samples) * 10
    samples = PointSet(coordinates=sample_coords)

    # Create target points
    target_coords = np.random.rand(n_targets, 2) * 1000
    targets = PointSet(coordinates=target_coords)

    # Create variogram model
    variogram = VariogramModel(
        model_type="spherical",
        nugget=0.1,
        sill=2.0,
        range_param=100.0,
        partial_sill=1.9,
        r_squared=0.95,
    )

    # Benchmark SGS
    start = time.perf_counter()
    realizations = sequential_gaussian_simulation(
        samples,
        sample_values,
        targets,
        variogram,
        n_realizations=n_realizations,
        random_seed=42,
    )
    total_time = time.perf_counter() - start

    return {
        "n_samples": n_samples,
        "n_targets": n_targets,
        "n_realizations": n_realizations,
        "total_time_seconds": total_time,
        "time_per_realization": total_time / n_realizations,
        "targets_per_second": (n_targets * n_realizations) / total_time,
    }


def benchmark_sgs_scalability() -> Dict[str, Dict[str, float]]:
    """Benchmark SGS across different problem sizes.

    Returns:
        Dictionary with results for different sizes.
    """
    results = {}

    configs = [
        ("small", 30, 500, 5),
        ("medium", 50, 1000, 10),
        ("large", 100, 2000, 10),
    ]

    for size_name, n_samples, n_targets, n_realizations in configs:
        print(f"  Benchmarking {size_name} ({n_samples} samples, {n_targets} targets, {n_realizations} realizations)...")
        results[size_name] = benchmark_sgs(
            n_samples=n_samples,
            n_targets=n_targets,
            n_realizations=n_realizations,
        )

    return results


def run_all_simulation_benchmarks() -> Dict[str, Dict]:
    """Run all simulation benchmarks and return results.

    Returns:
        Dictionary with all benchmark results.
    """
    results = {}

    print("Benchmarking Sequential Gaussian Simulation scalability...")
    results["sgs_scalability"] = benchmark_sgs_scalability()

    return results


if __name__ == "__main__":
    """Run benchmarks and print results."""
    results = run_all_simulation_benchmarks()

    print("\n" + "=" * 60)
    print("SIMULATION PERFORMANCE BENCHMARKS")
    print("=" * 60)

    print("\nSequential Gaussian Simulation Scalability:")
    for size, data in results["sgs_scalability"].items():
        print(f"  {size:8s}: {data['n_samples']:3d} samples, {data['n_targets']:5d} targets, {data['n_realizations']:2d} realizations")
        print(f"            Total time: {data['total_time_seconds']:6.2f} s")
        print(f"            Time per realization: {data['time_per_realization']:6.2f} s")
        print(f"            Throughput: {data['targets_per_second']:8.0f} targets/s")

