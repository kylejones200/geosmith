"""Example: Surrogate Models for Fast Simulation Emulation.

Demonstrates training ML emulators on expensive simulations to achieve
100x-1000x speedup while maintaining reasonable accuracy.

This is a key GenAI concept: using foundation models (ML) to create
fast surrogates for computationally expensive geoscience simulations.
"""

import sys
import time
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from geosmith import PointSet
    from geosmith.primitives.simulation import sequential_gaussian_simulation
    from geosmith.primitives.surrogate import SurrogateModel, train_simulation_emulator
    from geosmith.primitives.variogram import VariogramModel, fit_variogram_model
    from geosmith.tasks.surrogatetask import SurrogateTask

    SURROGATE_AVAILABLE = True
except ImportError:
    print("❌ Surrogate model dependencies not available.")
    print("Install with: pip install geosmith[genai]")
    print("Or: pip install scikit-learn xgboost")
    SURROGATE_AVAILABLE = False


def main():
    """Run surrogate model example."""
    if not SURROGATE_AVAILABLE:
        return

    print("=" * 70)
    print("Surrogate Models for Fast Simulation Emulation")
    print("=" * 70)
    print("\nGenAI Concept: Train ML models on expensive simulation I/O")
    print("to create fast emulators (100x-1000x speedup)\n")

    # Create synthetic data
    print("1. Creating synthetic sample data...")
    np.random.seed(42)
    n_samples = 100
    n_query = 1000

    # Sample locations
    sample_coords = np.random.rand(n_samples, 2) * 1000
    sample_values = (
        sample_coords[:, 0] * 0.1
        + sample_coords[:, 1] * 0.15
        + np.sin(sample_coords[:, 0] / 100) * 10
        + np.random.randn(n_samples) * 5
    )

    samples = PointSet(coordinates=sample_coords)

    # Query points (grid)
    query_x = np.linspace(0, 1000, int(np.sqrt(n_query)))
    query_y = np.linspace(0, 1000, int(np.sqrt(n_query)))
    query_xx, query_yy = np.meshgrid(query_x, query_y)
    query_coords = np.column_stack([query_xx.ravel(), query_yy.ravel()])
    query_points = PointSet(coordinates=query_coords)

    print(f"   Samples: {len(samples.coordinates)}")
    print(f"   Query points: {len(query_points.coordinates)}")

    # Create variogram model
    print("\n2. Fitting variogram model...")
    from geosmith.primitives.variogram import compute_experimental_variogram

    lags, semi_vars, n_pairs = compute_experimental_variogram(
        samples, sample_values, n_lags=15
    )
    variogram = fit_variogram_model(lags, semi_vars, model_type="spherical")
    print(f"   Variogram: {variogram.model_type}, range={variogram.range_param:.1f}")

    # Run expensive simulation (baseline)
    print("\n3. Running expensive simulation (baseline)...")
    start_time = time.time()
    sim_results = sequential_gaussian_simulation(
        samples,
        sample_values,
        query_points,
        variogram,
        n_realizations=1,
        random_seed=42,
    )
    sim_time = time.time() - start_time
    print(f"   Simulation time: {sim_time:.3f} seconds")
    print(f"   Results shape: {sim_results.shape}")

    # Train surrogate model
    print("\n4. Training surrogate model (XGBoost)...")
    print("   This is a one-time cost - surrogate can be reused many times")

    # Prepare training data
    # For SGS, inputs are [samples, query_points], output is sim_results
    training_inputs = [samples, query_points]
    training_outputs = sim_results.ravel()

    surrogate = train_simulation_emulator(
        simulation_func=sequential_gaussian_simulation,
        training_inputs=training_inputs,
        training_outputs=training_outputs,
        model_type="xgboost",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        input_params={
            "variogram_range": variogram.range_param,
            "nugget": variogram.nugget,
            "sill": variogram.sill,
        },
    )

    print(f"   Training time: {surrogate.training_time:.3f} seconds")
    if surrogate.metrics:
        print(f"   Validation R²: {surrogate.metrics.r2_score:.4f}")
        print(f"   Validation MAE: {surrogate.metrics.mae:.4f}")

    # Fast prediction with surrogate
    print("\n5. Fast prediction with surrogate model...")
    start_time = time.time()
    surrogate_predictions = surrogate.predict(query_points)
    surrogate_time = time.time() - start_time

    print(f"   Surrogate time: {surrogate_time:.6f} seconds")
    print(f"   Speedup: {sim_time / surrogate_time:.0f}x faster!")
    print(f"   Predictions shape: {surrogate_predictions.shape}")

    # Compare accuracy
    print("\n6. Comparing accuracy...")
    mae = np.mean(np.abs(sim_results.ravel() - surrogate_predictions))
    rmse = np.sqrt(np.mean((sim_results.ravel() - surrogate_predictions) ** 2))
    r2 = 1 - np.sum((sim_results.ravel() - surrogate_predictions) ** 2) / np.sum(
        (sim_results.ravel() - sim_results.ravel().mean()) ** 2
    )

    print(f"   MAE: {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   R²: {r2:.4f}")

    # Demonstrate reusability
    print("\n7. Demonstrating reusability...")
    print("   Surrogate can predict on new query points instantly:")

    # New query points
    new_query_coords = np.random.rand(500, 2) * 1000
    new_query_points = PointSet(coordinates=new_query_coords)

    start_time = time.time()
    new_predictions = surrogate.predict(new_query_points)
    new_time = time.time() - start_time

    print(f"   New predictions ({len(new_predictions)} points): {new_time:.6f} seconds")
    print(f"   (Full simulation would take ~{sim_time * len(new_predictions) / len(query_points.coordinates):.2f} seconds)")

    # Using Task layer
    print("\n8. Using Task layer (user-friendly interface)...")
    task = SurrogateTask(model_type="xgboost")
    task_predictions = task.predict(surrogate, new_query_points)
    print(f"   Task predictions: {len(task_predictions)} values")

    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"✅ Surrogate model trained successfully")
    print(f"✅ {sim_time / surrogate_time:.0f}x speedup achieved")
    print(f"✅ R² accuracy: {r2:.4f}")
    print(f"✅ Surrogate can be reused for new query points instantly")
    print("\n💡 Key Insight: Train once, predict many times!")
    print("   Perfect for iterative workflows, optimization, uncertainty analysis.")
    print("=" * 70)


if __name__ == "__main__":
    main()


