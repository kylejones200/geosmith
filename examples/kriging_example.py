"""Example: Kriging interpolation workflow.

Demonstrates variogram analysis and kriging interpolation
using GeoSmith's primitives.
"""

import numpy as np

from geosmith import PointSet
from geosmith.primitives.kriging import OrdinaryKriging
from geosmith.primitives.variogram import (
    compute_experimental_variogram,
    fit_variogram_model,
)


def main():
    """Run kriging example."""
    print("=" * 60)
    print("Kriging Interpolation Example")
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
    print(f"Computed variogram with {len(lags)} lag bins")
    print(f"Lag range: {lags[0]:.1f} - {lags[-1]:.1f}")

    # Fit variogram model
    print("\n3. Fitting variogram model...")
    variogram_model = fit_variogram_model(
        lags, semi_vars, model_type="spherical"
    )
    print(f"Fitted {variogram_model.model_type} model:")
    print(f"  Nugget: {variogram_model.nugget:.4f}")
    print(f"  Sill: {variogram_model.sill:.4f}")
    print(f"  Range: {variogram_model.range_param:.4f}")
    print(f"  R²: {variogram_model.r_squared:.4f}")

    # Fit kriging
    print("\n4. Fitting kriging model...")
    kriging = OrdinaryKriging(variogram_model=variogram_model)
    kriging.fit(samples, values)
    print("Kriging model fitted successfully")

    # Create query points
    print("\n5. Creating query points for prediction...")
    query_x = np.linspace(0, 1000, 20)
    query_y = np.linspace(0, 1000, 20)
    query_xx, query_yy = np.meshgrid(query_x, query_y)
    query_coords = np.column_stack([query_xx.ravel(), query_yy.ravel()])
    query_points = PointSet(coordinates=query_coords)
    print(f"Created {len(query_points.coordinates)} query points")

    # Predict
    print("\n6. Performing kriging prediction...")
    result = kriging.predict(query_points, return_variance=True)
    print(f"Predictions: mean={result.predictions.mean():.2f}, "
          f"std={result.predictions.std():.2f}")
    print(f"Variance: mean={result.variance.mean():.2f}, "
          f"std={result.variance.std():.2f}")

    # Display sample predictions
    print("\n7. Sample predictions:")
    for i in range(0, min(5, len(result.predictions)), 5):
        x_pred, y_pred = query_coords[i]
        pred = result.predictions[i]
        var = result.variance[i]
        print(f"  ({x_pred:.1f}, {y_pred:.1f}): "
              f"pred={pred:.2f}, var={var:.2f}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()


