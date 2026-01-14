"""Performance benchmarks for kriging operations."""

import time
from typing import Dict

import numpy as np

from geosmith import PointSet
from geosmith.primitives.kriging import (
    CoKriging,
    IndicatorKriging,
    OrdinaryKriging,
    SimpleKriging,
    UniversalKriging,
)
from geosmith.primitives.variogram import VariogramModel


def benchmark_ordinary_kriging(
    n_samples: int = 100,
    n_targets: int = 1000,
    n_dims: int = 3,
) -> Dict[str, float]:
    """Benchmark Ordinary Kriging performance.

    Args:
        n_samples: Number of sample points.
        n_targets: Number of target points to predict.
        n_dims: Number of dimensions (2 or 3).

    Returns:
        Dictionary with timing results.
    """
    np.random.seed(42)

    # Create sample data
    sample_coords = np.random.rand(n_samples, n_dims) * 1000
    sample_values = np.random.rand(n_samples) * 10
    samples = PointSet(coordinates=sample_coords)

    # Create target points
    target_coords = np.random.rand(n_targets, n_dims) * 1000
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

    # Benchmark fit
    kriging = OrdinaryKriging(variogram_model=variogram)
    start = time.perf_counter()
    kriging.fit(samples, sample_values)
    fit_time = time.perf_counter() - start

    # Benchmark predict
    start = time.perf_counter()
    result = kriging.predict(targets, return_variance=True)
    predict_time = time.perf_counter() - start

    return {
        "n_samples": n_samples,
        "n_targets": n_targets,
        "n_dims": n_dims,
        "fit_time_seconds": fit_time,
        "predict_time_seconds": predict_time,
        "predictions_per_second": n_targets / predict_time,
    }


def benchmark_kriging_types(
    n_samples: int = 50,
    n_targets: int = 500,
) -> Dict[str, Dict[str, float]]:
    """Benchmark different kriging types.

    Args:
        n_samples: Number of sample points.
        n_targets: Number of target points.

    Returns:
        Dictionary with results for each kriging type.
    """
    np.random.seed(42)

    sample_coords = np.random.rand(n_samples, 2) * 1000
    sample_values = np.random.rand(n_samples) * 10
    samples = PointSet(coordinates=sample_coords)

    target_coords = np.random.rand(n_targets, 2) * 1000
    targets = PointSet(coordinates=target_coords)

    variogram = VariogramModel(
        model_type="spherical",
        nugget=0.1,
        sill=2.0,
        range_param=100.0,
        partial_sill=1.9,
        r_squared=0.95,
    )

    results = {}

    # Ordinary Kriging
    kriging = OrdinaryKriging(variogram_model=variogram)
    start = time.perf_counter()
    kriging.fit(samples, sample_values)
    fit_time = time.perf_counter() - start
    start = time.perf_counter()
    kriging.predict(targets)
    predict_time = time.perf_counter() - start
    results["ordinary"] = {
        "fit_time": fit_time,
        "predict_time": predict_time,
    }

    # Simple Kriging
    kriging = SimpleKriging(variogram_model=variogram, mean=np.mean(sample_values))
    start = time.perf_counter()
    kriging.fit(samples, sample_values)
    fit_time = time.perf_counter() - start
    start = time.perf_counter()
    kriging.predict(targets)
    predict_time = time.perf_counter() - start
    results["simple"] = {
        "fit_time": fit_time,
        "predict_time": predict_time,
    }

    # Universal Kriging
    kriging = UniversalKriging(variogram_model=variogram, drift_terms=["linear"])
    start = time.perf_counter()
    kriging.fit(samples, sample_values)
    fit_time = time.perf_counter() - start
    start = time.perf_counter()
    kriging.predict(targets)
    predict_time = time.perf_counter() - start
    results["universal"] = {
        "fit_time": fit_time,
        "predict_time": predict_time,
    }

    # Indicator Kriging
    kriging = IndicatorKriging(variogram_model=variogram, threshold=5.0)
    start = time.perf_counter()
    kriging.fit(samples, sample_values)
    fit_time = time.perf_counter() - start
    start = time.perf_counter()
    kriging.predict(targets)
    predict_time = time.perf_counter() - start
    results["indicator"] = {
        "fit_time": fit_time,
        "predict_time": predict_time,
    }

    return results


def benchmark_co_kriging(
    n_primary: int = 30,
    n_secondary: int = 50,
    n_targets: int = 500,
) -> Dict[str, float]:
    """Benchmark Co-Kriging performance.

    Args:
        n_primary: Number of primary variable samples.
        n_secondary: Number of secondary variable samples.
        n_targets: Number of target points.

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

    target_coords = np.random.rand(n_targets, 2) * 1000
    targets = PointSet(coordinates=target_coords)

    # Create variogram models
    variogram = VariogramModel(
        model_type="spherical",
        nugget=0.1,
        sill=2.0,
        range_param=100.0,
        partial_sill=1.9,
        r_squared=0.95,
    )

    # Benchmark fit
    co_kriging = CoKriging(
        primary_variogram=variogram,
        secondary_variogram=variogram,
        cross_variogram=variogram,
    )
    start = time.perf_counter()
    co_kriging.fit(primary_points, primary_values, secondary_points, secondary_values)
    fit_time = time.perf_counter() - start

    # Benchmark predict
    start = time.perf_counter()
    result = co_kriging.predict(targets, return_variance=True)
    predict_time = time.perf_counter() - start

    return {
        "n_primary": n_primary,
        "n_secondary": n_secondary,
        "n_targets": n_targets,
        "fit_time_seconds": fit_time,
        "predict_time_seconds": predict_time,
        "predictions_per_second": n_targets / predict_time,
    }


def run_all_kriging_benchmarks() -> Dict[str, Dict]:
    """Run all kriging benchmarks and return results.

    Returns:
        Dictionary with all benchmark results.
    """
    results = {}

    # Ordinary Kriging scalability
    print("Benchmarking Ordinary Kriging scalability...")
    results["ordinary_scalability"] = {
        "small": benchmark_ordinary_kriging(50, 500, 2),
        "medium": benchmark_ordinary_kriging(200, 2000, 2),
        "large": benchmark_ordinary_kriging(500, 5000, 2),
        "3d": benchmark_ordinary_kriging(100, 1000, 3),
    }

    # Kriging type comparison
    print("Benchmarking different kriging types...")
    results["kriging_types"] = benchmark_kriging_types(50, 500)

    # Co-Kriging
    print("Benchmarking Co-Kriging...")
    results["co_kriging"] = benchmark_co_kriging(30, 50, 500)

    return results


if __name__ == "__main__":
    """Run benchmarks and print results."""
    results = run_all_kriging_benchmarks()

    print("\n" + "=" * 60)
    print("KRIGING PERFORMANCE BENCHMARKS")
    print("=" * 60)

    print("\nOrdinary Kriging Scalability:")
    for size, data in results["ordinary_scalability"].items():
        print(f"  {size:8s}: {data['n_samples']:4d} samples, {data['n_targets']:5d} targets")
        print(f"            Fit: {data['fit_time_seconds']*1000:6.2f} ms")
        print(f"            Predict: {data['predict_time_seconds']*1000:6.2f} ms")
        print(f"            Throughput: {data['predictions_per_second']:8.0f} pred/s")

    print("\nKriging Type Comparison (50 samples, 500 targets):")
    for ktype, data in results["kriging_types"].items():
        print(f"  {ktype:10s}: Fit {data['fit_time']*1000:6.2f} ms, "
              f"Predict {data['predict_time']*1000:6.2f} ms")

    print("\nCo-Kriging (30 primary, 50 secondary, 500 targets):")
    ck = results["co_kriging"]
    print(f"  Fit: {ck['fit_time_seconds']*1000:6.2f} ms")
    print(f"  Predict: {ck['predict_time_seconds']*1000:6.2f} ms")
    print(f"  Throughput: {ck['predictions_per_second']:8.0f} pred/s")

