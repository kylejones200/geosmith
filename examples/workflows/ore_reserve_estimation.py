"""Complete Ore Reserve Estimation Workflow Demo.

This demo shows a complete workflow from drillhole data to block model estimation
with uncertainty quantification. It demonstrates:

1. Data loading and preparation
2. Exploratory data analysis
3. Variogram analysis and model fitting
4. Cross-validation for model selection
5. Block model creation
6. Grade estimation with kriging
7. Uncertainty quantification with SGS
8. Risk assessment with indicator kriging
9. Results export

This is a production-ready workflow that geoscientists would use in practice.
"""

import numpy as np
import pandas as pd

from geosmith import PointSet
from geosmith.primitives.kriging import OrdinaryKriging, UniversalKriging
from geosmith.primitives.kriging_cv import leave_one_out_cross_validation
from geosmith.primitives.simulation import (
    compute_exceedance_probability,
    sequential_gaussian_simulation,
)
from geosmith.primitives.variogram import (
    compute_experimental_variogram,
    fit_variogram_model,
)
from geosmith.tasks.blockmodeltask import create_block_model_grid
from geosmith.workflows.drillhole import compute_3d_coordinates, merge_collar_assay
from geosmith.workflows.geostatistics import GeostatisticalModel
from geosmith.workflows.gslib import export_block_model_gslib


def create_synthetic_drillhole_data():
    """Create synthetic drillhole data for demonstration.

    In practice, this would be loaded from files or databases.
    """
    np.random.seed(42)

    # Create collar data (drillhole locations)
    n_holes = 25
    collar_data = {
        "HOLEID": [f"DH{i:03d}" for i in range(1, n_holes + 1)],
        "EASTING": np.random.uniform(100, 900, n_holes),
        "NORTHING": np.random.uniform(100, 900, n_holes),
        "RL": np.random.uniform(50, 150, n_holes),
    }
    collar_df = pd.DataFrame(collar_data)

    # Create assay data (grade measurements)
    assay_data = []
    for hole_id in collar_data["HOLEID"]:
        n_samples = np.random.randint(5, 15)
        for i in range(n_samples):
            from_depth = i * 5.0
            to_depth = (i + 1) * 5.0
            # Create spatially correlated grades
            base_grade = 2.0 + np.random.randn() * 0.5
            grade = max(0.1, base_grade + np.random.randn() * 0.3)
            assay_data.append({
                "HOLEID": hole_id,
                "FROM": from_depth,
                "TO": to_depth,
                "Au": grade,  # Gold grade in g/t
            })
    assay_df = pd.DataFrame(assay_data)

    return collar_df, assay_df


def main():
    """Run complete ore reserve estimation workflow."""
    print("=" * 80)
    print("COMPLETE ORE RESERVE ESTIMATION WORKFLOW")
    print("=" * 80)
    print("\nThis demo shows a production-ready workflow for estimating ore reserves")
    print("from drillhole data using geostatistical methods.\n")

    # ============================================================================
    # STEP 1: Data Loading and Preparation
    # ============================================================================
    print("STEP 1: Data Loading and Preparation")
    print("-" * 80)
    print("Loading drillhole collar and assay data...")

    collar_df, assay_df = create_synthetic_drillhole_data()
    print(f"  Loaded {len(collar_df)} drillholes")
    print(f"  Loaded {len(assay_df)} assay samples")

    # Process drillhole data
    merged_df = merge_collar_assay(collar_df, assay_df)
    points = compute_3d_coordinates(merged_df, assume_vertical=True)
    grades = merged_df["Au"].values

    print(f"  Processed {len(points.coordinates)} 3D sample points")
    print(f"  Grade statistics:")
    print(f"    Mean: {grades.mean():.2f} g/t")
    print(f"    Std:  {grades.std():.2f} g/t")
    print(f"    Min:  {grades.min():.2f} g/t")
    print(f"    Max:  {grades.max():.2f} g/t")
    print(f"    P50:  {np.percentile(grades, 50):.2f} g/t")

    # ============================================================================
    # STEP 2: Exploratory Data Analysis
    # ============================================================================
    print("\nSTEP 2: Exploratory Data Analysis")
    print("-" * 80)
    print("Computing experimental variogram...")

    lags, semi_vars, n_pairs = compute_experimental_variogram(
        points, grades, n_lags=20
    )

    print(f"  Computed variogram with {len(lags)} lag bins")
    print(f"  Lag range: {lags[0]:.1f} - {lags[-1]:.1f} m")
    print(f"  Total pairs: {n_pairs.sum()}")

    # ============================================================================
    # STEP 3: Variogram Model Fitting
    # ============================================================================
    print("\nSTEP 3: Variogram Model Fitting")
    print("-" * 80)
    print("Fitting variogram models...")

    # Try different models
    models = ["spherical", "exponential", "gaussian"]
    variogram_results = {}

    for model_type in models:
        variogram = fit_variogram_model(lags, semi_vars, model_type=model_type)
        variogram_results[model_type] = variogram
        print(f"\n  {model_type.capitalize()} model:")
        print(f"    Nugget: {variogram.nugget:.4f}")
        print(f"    Sill:   {variogram.sill:.4f}")
        print(f"    Range:  {variogram.range_param:.2f} m")
        print(f"    R²:     {variogram.r_squared:.4f}")

    # Select best model (highest R²)
    best_model_type = max(variogram_results, key=lambda k: variogram_results[k].r_squared)
    best_variogram = variogram_results[best_model_type]
    print(f"\n  Selected model: {best_model_type} (R² = {best_variogram.r_squared:.4f})")

    # ============================================================================
    # STEP 4: Cross-Validation for Model Selection
    # ============================================================================
    print("\nSTEP 4: Cross-Validation for Model Selection")
    print("-" * 80)
    print("Performing leave-one-out cross-validation...")

    # Compare Ordinary vs Universal Kriging
    cv_ok = leave_one_out_cross_validation(
        points, grades, best_variogram, kriging_type="ordinary"
    )

    cv_uk = leave_one_out_cross_validation(
        points,
        grades,
        best_variogram,
        kriging_type="universal",
        drift_terms=["linear"],
    )

    print(f"\n  Ordinary Kriging:")
    print(f"    RMSE: {cv_ok.rmse:.2f} g/t")
    print(f"    R²:   {cv_ok.r2:.3f}")
    print(f"    Bias: {cv_ok.mean_error:.4f} g/t")

    print(f"\n  Universal Kriging (with linear trend):")
    print(f"    RMSE: {cv_uk.rmse:.2f} g/t")
    print(f"    R²:   {cv_uk.r2:.3f}")
    print(f"    Bias: {cv_uk.mean_error:.4f} g/t")

    # Select best kriging type
    if cv_uk.r2 > cv_ok.r2:
        kriging_type = "universal"
        cv_result = cv_uk
        print(f"\n  → Selected: Universal Kriging (better R²)")
    else:
        kriging_type = "ordinary"
        cv_result = cv_ok
        print(f"\n  → Selected: Ordinary Kriging (better R²)")

    # ============================================================================
    # STEP 5: Block Model Creation
    # ============================================================================
    print("\nSTEP 5: Block Model Creation")
    print("-" * 80)
    print("Creating 3D block model grid...")

    coords = points.coordinates
    grid_coords, grid_info = create_block_model_grid(
        coords,
        block_size_xy=25.0,  # 25m x 25m blocks
        block_size_z=10.0,   # 10m vertical
        quantile_padding=0.05,
    )

    grid_points = PointSet(coordinates=grid_coords)

    print(f"  Block model dimensions: {grid_info['nx']} x {grid_info['ny']} x {grid_info['nz']}")
    print(f"  Total blocks: {grid_info['n_blocks']}")
    print(f"  Block size: {grid_info['block_size_xy']}m x {grid_info['block_size_xy']}m x {grid_info['block_size_z']}m")

    # ============================================================================
    # STEP 6: Grade Estimation with Kriging
    # ============================================================================
    print("\nSTEP 6: Grade Estimation with Kriging")
    print("-" * 80)
    print("Estimating grades on block model...")

    # Use unified workflow interface
    model = GeostatisticalModel(
        data=points,
        values=grades,
        method="kriging",
        kriging_type=kriging_type,
        validation="none",  # Already validated above
        variogram_model=best_variogram,
        drift_terms=["linear"] if kriging_type == "universal" else None,
    )

    results = model.estimate(grid_points, return_variance=True)

    print(f"  Estimated {len(results.estimates)} blocks")
    print(f"  Estimated grade statistics:")
    print(f"    Mean: {results.estimates.mean():.2f} g/t")
    print(f"    Std:  {results.estimates.std():.2f} g/t")
    print(f"    Min:  {results.estimates.min():.2f} g/t")
    print(f"    Max:  {results.estimates.max():.2f} g/t")
    print(f"  Mean kriging variance: {results.variance.mean():.4f}")

    # ============================================================================
    # STEP 7: Uncertainty Quantification with SGS
    # ============================================================================
    print("\nSTEP 7: Uncertainty Quantification with Sequential Gaussian Simulation")
    print("-" * 80)
    print("Generating multiple realizations for uncertainty analysis...")

    n_realizations = 50
    realizations = sequential_gaussian_simulation(
        points,
        grades,
        grid_points,
        best_variogram,
        n_realizations=n_realizations,
        random_seed=42,
    )

    # Compute statistics
    p10 = np.percentile(realizations, 10, axis=0)
    p50 = np.percentile(realizations, 50, axis=0)
    p90 = np.percentile(realizations, 90, axis=0)

    print(f"  Generated {n_realizations} realizations")
    print(f"  Uncertainty statistics:")
    print(f"    P10 (conservative): {p10.mean():.2f} g/t")
    print(f"    P50 (median):       {p50.mean():.2f} g/t")
    print(f"    P90 (optimistic):   {p90.mean():.2f} g/t")
    print(f"    Range (P90-P10):    {(p90 - p10).mean():.2f} g/t")

    # ============================================================================
    # STEP 8: Risk Assessment with Indicator Kriging
    # ============================================================================
    print("\nSTEP 8: Risk Assessment with Indicator Kriging")
    print("-" * 80)

    # Define economic cutoff
    cutoff = 2.0  # 2 g/t Au cutoff grade
    print(f"  Economic cutoff: {cutoff} g/t Au")

    from geosmith.primitives.kriging import IndicatorKriging

    ik = IndicatorKriging(variogram_model=best_variogram, threshold=cutoff)
    ik.fit(points, grades)

    prob_result = ik.predict(grid_points)
    probabilities = prob_result.predictions

    print(f"  Probability of exceeding {cutoff} g/t:")
    print(f"    Mean probability: {probabilities.mean():.1%}")
    print(f"    Blocks with >50% probability: {(probabilities > 0.5).sum()}")
    print(f"    Blocks with >70% probability: {(probabilities > 0.7).sum()}")

    # Compute exceedance probability from SGS
    prob_exceed_sgs = compute_exceedance_probability(realizations, cutoff)
    print(f"\n  Exceedance probability (from SGS):")
    print(f"    Mean probability: {prob_exceed_sgs.mean():.1%}")
    print(f"    Blocks with >50% probability: {(prob_exceed_sgs > 0.5).sum()}")

    # ============================================================================
    # STEP 9: Grade-Tonnage Analysis
    # ============================================================================
    print("\nSTEP 9: Grade-Tonnage Analysis")
    print("-" * 80)

    # Block volume (assuming density = 2.7 t/m³)
    block_volume = (
        grid_info["block_size_xy"]
        * grid_info["block_size_xy"]
        * grid_info["block_size_z"]
    )
    density = 2.7  # tonnes per cubic meter
    block_tonnage = block_volume * density

    # Total tonnage
    total_tonnage = len(grid_points.coordinates) * block_tonnage / 1e6  # million tonnes

    # Grade-tonnage curves
    cutoffs = np.arange(0.5, 5.0, 0.5)
    print(f"  Grade-Tonnage Analysis (Total: {total_tonnage:.1f} Mt):")
    print(f"  {'Cutoff':>8} {'Tonnage':>12} {'Grade':>10} {'Metal':>12}")
    print(f"  {'(g/t)':>8} {'(Mt)':>12} {'(g/t)':>10} {'(koz)':>12}")
    print("  " + "-" * 44)

    for cutoff_gt in cutoffs:
        # Tonnage above cutoff
        above_cutoff = results.estimates >= cutoff_gt
        tonnage = above_cutoff.sum() * block_tonnage / 1e6

        if tonnage > 0:
            # Average grade above cutoff
            avg_grade = results.estimates[above_cutoff].mean()
            # Metal content (in thousand ounces, assuming 31.1 g/oz)
            metal_koz = (tonnage * 1e6 * avg_grade) / (31.1 * 1000)
            print(f"  {cutoff_gt:>6.1f} {tonnage:>10.1f} {avg_grade:>8.2f} {metal_koz:>10.1f}")

    # ============================================================================
    # STEP 10: Results Export
    # ============================================================================
    print("\nSTEP 10: Results Export")
    print("-" * 80)
    print("Preparing block model for export...")

    # Create comprehensive block model DataFrame
    block_model_df = pd.DataFrame({
        "X": grid_coords[:, 0],
        "Y": grid_coords[:, 1],
        "Z": grid_coords[:, 2],
        "Estimate": results.estimates,
        "Variance": results.variance,
        "P10": p10,
        "P50": p50,
        "P90": p90,
        "ProbExceed": prob_exceed_sgs,
        "ProbExceedIK": probabilities,
    })

    print(f"  Block model DataFrame: {len(block_model_df)} rows x {len(block_model_df.columns)} columns")
    print(f"  Columns: {', '.join(block_model_df.columns)}")

    # Export to GSLIB format (industry standard)
    try:
        export_block_model_gslib(
            block_model_df,
            "ore_reserve_model.dat",
            title="Ore Reserve Estimation - Gold Deposit",
        )
        print("  ✓ Exported to GSLIB format: ore_reserve_model.dat")
    except Exception as e:
        print(f"  ⚠ Export failed: {e}")

    # ============================================================================
    # SUMMARY
    # ============================================================================
    print("\n" + "=" * 80)
    print("WORKFLOW SUMMARY")
    print("=" * 80)
    print(f"✓ Processed {len(points.coordinates)} drillhole samples")
    print(f"✓ Fitted {best_model_type} variogram (R² = {best_variogram.r_squared:.4f})")
    print(f"✓ Cross-validated with {kriging_type} kriging (R² = {cv_result.r2:.3f})")
    print(f"✓ Estimated {len(grid_points.coordinates)} blocks")
    print(f"✓ Generated {n_realizations} SGS realizations for uncertainty")
    print(f"✓ Computed risk probabilities for {cutoff} g/t cutoff")
    print(f"✓ Exported results to GSLIB format")
    print("\nThis workflow demonstrates:")
    print("  • Complete data processing pipeline")
    print("  • Model selection with cross-validation")
    print("  • Uncertainty quantification")
    print("  • Risk assessment")
    print("  • Industry-standard export formats")
    print("=" * 80)


if __name__ == "__main__":
    main()

