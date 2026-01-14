"""Reservoir Property Modeling Workflow Demo.

This demo shows a complete workflow for modeling reservoir properties from well logs:

1. Load well log data
2. Petrophysical property calculation
3. Spatial property modeling with kriging
4. Uncertainty quantification
5. Property maps generation

This workflow is used in petroleum reservoir characterization.
"""

import numpy as np
import pandas as pd

from geosmith import PointSet
from geosmith.primitives.kriging import OrdinaryKriging
from geosmith.primitives.kriging_cv import leave_one_out_cross_validation
from geosmith.primitives.petrophysics import (
    ArchieParams,
    calculate_permeability_timur,
    calculate_porosity_from_density,
    calculate_water_saturation,
)
from geosmith.primitives.simulation import sequential_gaussian_simulation
from geosmith.primitives.variogram import (
    compute_experimental_variogram,
    fit_variogram_model,
)
from geosmith.workflows.geostatistics import GeostatisticalModel


def create_synthetic_well_data():
    """Create synthetic well log data for demonstration."""
    np.random.seed(42)

    # Well locations
    n_wells = 20
    well_coords = np.random.rand(n_wells, 2) * 5000  # 5km x 5km field

    # Well log data (depth-based measurements)
    well_data = []
    for i, (x, y) in enumerate(well_coords):
        well_id = f"WELL_{i+1:03d}"
        n_depths = np.random.randint(50, 100)

        for depth in np.linspace(2000, 3000, n_depths):  # 2000-3000m depth
            # Create correlated petrophysical properties
            # Porosity (typically 0.1-0.3)
            porosity = 0.15 + 0.1 * np.random.rand() + np.random.randn() * 0.02
            porosity = np.clip(porosity, 0.05, 0.35)

            # Density (g/cc) - related to porosity
            density = 2.65 - 1.5 * porosity + np.random.randn() * 0.05

            # Resistivity (ohm-m) - related to water saturation
            rt = 5.0 + 20 * porosity + np.random.randn() * 2
            rt = np.maximum(1.0, rt)

            well_data.append({
                "WELL_ID": well_id,
                "X": x,
                "Y": y,
                "DEPTH": depth,
                "DENSITY": density,
                "RESISTIVITY": rt,
            })

    return pd.DataFrame(well_data), well_coords


def main():
    """Run reservoir property modeling workflow."""
    print("=" * 80)
    print("RESERVOIR PROPERTY MODELING WORKFLOW")
    print("=" * 80)
    print("\nThis demo shows how to model reservoir properties (porosity, permeability,")
    print("water saturation) from well logs using geostatistical methods.\n")

    # ============================================================================
    # STEP 1: Load Well Log Data
    # ============================================================================
    print("STEP 1: Load Well Log Data")
    print("-" * 80)

    well_df, well_coords = create_synthetic_well_data()
    print(f"  Loaded {len(well_df)} well log measurements from {well_df['WELL_ID'].nunique()} wells")
    print(f"  Depth range: {well_df['DEPTH'].min():.0f} - {well_df['DEPTH'].max():.0f} m")
    print(f"  Variables: Density, Resistivity")

    # ============================================================================
    # STEP 2: Calculate Petrophysical Properties
    # ============================================================================
    print("\nSTEP 2: Calculate Petrophysical Properties")
    print("-" * 80)
    print("Computing porosity, permeability, and water saturation...")

    # Porosity from density
    rho_matrix = 2.65  # Matrix density (g/cc)
    rho_fluid = 1.0    # Fluid density (g/cc)
    porosity = calculate_porosity_from_density(
        well_df["DENSITY"].values, rho_matrix, rho_fluid
    )

    # Permeability from porosity (Timur equation)
    permeability = calculate_permeability_timur(porosity)

    # Water saturation (Archie's equation)
    archie_params = ArchieParams(a=1.0, m=2.0, n=2.0, rw=0.1)
    water_saturation = calculate_water_saturation(
        well_df["RESISTIVITY"].values, porosity, archie_params
    )

    well_df["POROSITY"] = porosity
    well_df["PERMEABILITY"] = permeability
    well_df["WATER_SATURATION"] = water_saturation

    print(f"  Calculated properties:")
    print(f"    Porosity: {porosity.mean():.3f} ± {porosity.std():.3f}")
    print(f"    Permeability: {permeability.mean():.1f} ± {permeability.std():.1f} mD")
    print(f"    Water saturation: {water_saturation.mean():.1%} ± {water_saturation.std():.1%}")

    # ============================================================================
    # STEP 3: Create 3D Sample Points
    # ============================================================================
    print("\nSTEP 3: Create 3D Sample Points")
    print("-" * 80)

    # Use average depth for each well (or could use all depths)
    well_summary = well_df.groupby("WELL_ID").agg({
        "X": "first",
        "Y": "first",
        "POROSITY": "mean",
        "PERMEABILITY": "mean",
        "WATER_SATURATION": "mean",
    }).reset_index()

    # Create 3D points (X, Y, Depth)
    coords_3d = np.column_stack([
        well_summary["X"].values,
        well_summary["Y"].values,
        well_summary.groupby("WELL_ID")["DEPTH"].mean().values if "DEPTH" in well_df.columns
        else np.full(len(well_summary), 2500),  # Average depth
    ])

    points_3d = PointSet(coordinates=coords_3d)

    print(f"  Created {len(points_3d.coordinates)} 3D sample points")
    print(f"  Coordinate ranges:")
    print(f"    X: {coords_3d[:, 0].min():.0f} - {coords_3d[:, 0].max():.0f} m")
    print(f"    Y: {coords_3d[:, 1].min():.0f} - {coords_3d[:, 1].max():.0f} m")
    print(f"    Z: {coords_3d[:, 2].min():.0f} - {coords_3d[:, 2].max():.0f} m")

    # ============================================================================
    # STEP 4: Model Porosity with Kriging
    # ============================================================================
    print("\nSTEP 4: Model Porosity with Kriging")
    print("-" * 80)

    porosity_values = well_summary["POROSITY"].values

    # Compute variogram
    print("  Computing experimental variogram...")
    lags, semi_vars, _ = compute_experimental_variogram(
        points_3d, porosity_values, n_lags=15
    )

    # Fit variogram model
    variogram = fit_variogram_model(lags, semi_vars, model_type="spherical")
    print(f"  Fitted variogram:")
    print(f"    Model: {variogram.model_type}")
    print(f"    Nugget: {variogram.nugget:.4f}")
    print(f"    Sill: {variogram.sill:.4f}")
    print(f"    Range: {variogram.range_param:.0f} m")
    print(f"    R²: {variogram.r_squared:.4f}")

    # Cross-validation
    print("\n  Cross-validating model...")
    cv_result = leave_one_out_cross_validation(
        points_3d, porosity_values, variogram, kriging_type="ordinary"
    )
    print(f"    RMSE: {cv_result.rmse:.4f}")
    print(f"    R²: {cv_result.r2:.3f}")
    print(f"    Bias: {cv_result.mean_error:.4f}")

    # ============================================================================
    # STEP 5: Create 3D Grid and Estimate Properties
    # ============================================================================
    print("\nSTEP 5: Create 3D Grid and Estimate Properties")
    print("-" * 80)

    # Create query grid (2D map view at average depth)
    grid_x = np.linspace(0, 5000, 30)
    grid_y = np.linspace(0, 5000, 30)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
    grid_z = np.full_like(grid_xx, 2500)  # Constant depth

    grid_coords = np.column_stack([
        grid_xx.ravel(),
        grid_yy.ravel(),
        grid_z.ravel(),
    ])
    grid_points = PointSet(coordinates=grid_coords)

    print(f"  Created {len(grid_points.coordinates)} grid points")

    # Estimate porosity
    print("\n  Estimating porosity on grid...")
    model = GeostatisticalModel(
        data=points_3d,
        values=porosity_values,
        method="kriging",
        kriging_type="ordinary",
        variogram_model=variogram,
    )

    porosity_results = model.estimate(grid_points, return_variance=True)

    print(f"    Mean porosity: {porosity_results.estimates.mean():.3f}")
    print(f"    Std porosity: {porosity_results.estimates.std():.3f}")
    print(f"    Mean variance: {porosity_results.variance.mean():.4f}")

    # ============================================================================
    # STEP 6: Uncertainty Quantification
    # ============================================================================
    print("\nSTEP 6: Uncertainty Quantification")
    print("-" * 80)
    print("  Generating multiple realizations...")

    n_realizations = 30
    porosity_realizations = sequential_gaussian_simulation(
        points_3d,
        porosity_values,
        grid_points,
        variogram,
        n_realizations=n_realizations,
        random_seed=42,
    )

    porosity_p10 = np.percentile(porosity_realizations, 10, axis=0)
    porosity_p50 = np.percentile(porosity_realizations, 50, axis=0)
    porosity_p90 = np.percentile(porosity_realizations, 90, axis=0)

    print(f"    Generated {n_realizations} realizations")
    print(f"    P10: {porosity_p10.mean():.3f}")
    print(f"    P50: {porosity_p50.mean():.3f}")
    print(f"    P90: {porosity_p90.mean():.3f}")

    # ============================================================================
    # STEP 7: Model Other Properties
    # ============================================================================
    print("\nSTEP 7: Model Permeability and Water Saturation")
    print("-" * 80)

    # Permeability
    perm_values = well_summary["PERMEABILITY"].values
    lags_perm, semi_vars_perm, _ = compute_experimental_variogram(
        points_3d, perm_values, n_lags=15
    )
    variogram_perm = fit_variogram_model(lags_perm, semi_vars_perm, model_type="spherical")

    model_perm = GeostatisticalModel(
        data=points_3d,
        values=perm_values,
        method="kriging",
        kriging_type="ordinary",
        variogram_model=variogram_perm,
    )
    perm_results = model_perm.estimate(grid_points, return_variance=True)

    # Water saturation
    sw_values = well_summary["WATER_SATURATION"].values
    lags_sw, semi_vars_sw, _ = compute_experimental_variogram(
        points_3d, sw_values, n_lags=15
    )
    variogram_sw = fit_variogram_model(lags_sw, semi_vars_sw, model_type="spherical")

    model_sw = GeostatisticalModel(
        data=points_3d,
        values=sw_values,
        method="kriging",
        kriging_type="ordinary",
        variogram_model=variogram_sw,
    )
    sw_results = model_sw.estimate(grid_points, return_variance=True)

    print(f"  Permeability:")
    print(f"    Mean: {perm_results.estimates.mean():.1f} mD")
    print(f"    Std: {perm_results.estimates.std():.1f} mD")
    print(f"  Water saturation:")
    print(f"    Mean: {sw_results.estimates.mean():.1%}")
    print(f"    Std: {sw_results.estimates.std():.1%}")

    # ============================================================================
    # STEP 8: Results Summary
    # ============================================================================
    print("\n" + "=" * 80)
    print("WORKFLOW SUMMARY")
    print("=" * 80)
    print(f"✓ Processed {len(well_df)} well log measurements")
    print(f"✓ Calculated petrophysical properties for {len(points_3d.coordinates)} locations")
    print(f"✓ Modeled porosity, permeability, and water saturation")
    print(f"✓ Generated {n_realizations} uncertainty realizations")
    print(f"✓ Created property maps on {len(grid_points.coordinates)} grid points")
    print("\nThis workflow demonstrates:")
    print("  • Petrophysical property calculation from well logs")
    print("  • 3D spatial modeling with kriging")
    print("  • Uncertainty quantification with SGS")
    print("  • Multi-property reservoir characterization")
    print("=" * 80)

    # Create results DataFrame
    results_df = pd.DataFrame({
        "X": grid_coords[:, 0],
        "Y": grid_coords[:, 1],
        "Z": grid_coords[:, 2],
        "Porosity": porosity_results.estimates,
        "Porosity_Variance": porosity_results.variance,
        "Porosity_P10": porosity_p10,
        "Porosity_P50": porosity_p50,
        "Porosity_P90": porosity_p90,
        "Permeability": perm_results.estimates,
        "Permeability_Variance": perm_results.variance,
        "WaterSaturation": sw_results.estimates,
        "WaterSaturation_Variance": sw_results.variance,
    })

    return results_df


if __name__ == "__main__":
    results = main()
    print(f"\nResults DataFrame shape: {results.shape}")
    print(f"Columns: {', '.join(results.columns)}")

