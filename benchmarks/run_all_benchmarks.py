"""Run all performance benchmarks and generate report."""

import json
from pathlib import Path

from benchmark_kriging import run_all_kriging_benchmarks
from benchmark_simulation import run_all_simulation_benchmarks
from benchmark_variogram import run_all_variogram_benchmarks


def main():
    """Run all benchmarks and save results."""
    print("=" * 60)
    print("GEOSMITH PERFORMANCE BENCHMARK SUITE")
    print("=" * 60)
    print()

    all_results = {}

    # Run variogram benchmarks
    print("\n[1/3] Variogram Benchmarks")
    print("-" * 60)
    all_results["variogram"] = run_all_variogram_benchmarks()

    # Run kriging benchmarks
    print("\n[2/3] Kriging Benchmarks")
    print("-" * 60)
    all_results["kriging"] = run_all_kriging_benchmarks()

    # Run simulation benchmarks
    print("\n[3/3] Simulation Benchmarks")
    print("-" * 60)
    all_results["simulation"] = run_all_simulation_benchmarks()

    # Save results to JSON
    output_file = Path("benchmarks/results.json")
    output_file.parent.mkdir(exist_ok=True)

    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        """Convert numpy types to native Python types."""
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native(item) for item in obj]
        elif hasattr(obj, "item"):  # numpy scalar
            return obj.item()
        else:
            return obj

    json_results = convert_to_native(all_results)

    with open(output_file, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    # Variogram summary
    var_results = all_results["variogram"]["variogram_scalability"]
    print(f"\nVariogram Computation:")
    print(f"  Small (100 samples):   {var_results['small']['total_time_seconds']*1000:6.2f} ms")
    print(f"  Large (5000 samples):  {var_results['xlarge']['total_time_seconds']*1000:6.2f} ms")

    # Kriging summary
    krig_results = all_results["kriging"]["ordinary_scalability"]
    print(f"\nOrdinary Kriging (500 targets):")
    print(f"  Small (50 samples):     {krig_results['small']['predict_time_seconds']*1000:6.2f} ms")
    print(f"  Large (500 samples):   {krig_results['large']['predict_time_seconds']*1000:6.2f} ms")

    # Simulation summary
    sim_results = all_results["simulation"]["sgs_scalability"]
    print(f"\nSequential Gaussian Simulation:")
    print(f"  Small (500 targets, 5 realizations):  {sim_results['small']['total_time_seconds']:6.2f} s")
    print(f"  Large (2000 targets, 10 realizations): {sim_results['large']['total_time_seconds']:6.2f} s")

    print("\n✓ All benchmarks completed successfully!")


if __name__ == "__main__":
    main()

