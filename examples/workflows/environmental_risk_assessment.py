"""Environmental Risk Assessment Workflow Demo.

This demo shows how to assess environmental contamination risk using:

1. Sample point data loading
2. Spatial autocorrelation analysis
3. Contamination mapping with kriging
4. Risk assessment with indicator kriging
5. Exceedance probability mapping
6. Hotspot identification

This workflow is used for:
- Environmental site assessment
- Contamination monitoring
- Regulatory compliance
- Remediation planning
"""

import numpy as np
import pandas as pd

from geosmith import PointSet
from geosmith.primitives.kriging import IndicatorKriging
from geosmith.primitives.simulation import (
    compute_exceedance_probability,
    sequential_gaussian_simulation,
)
from geosmith.primitives.spatial_analysis import (
    create_knn_weights,
    getis_ord_gi_star,
    morans_i,
)
from geosmith.primitives.variogram import (
    compute_experimental_variogram,
    fit_variogram_model,
)
from geosmith.workflows.geostatistics import GeostatisticalModel


def create_synthetic_contamination_data():
    """Create synthetic contamination data for demonstration."""
    np.random.seed(42)

    # Create sample locations (monitoring wells, soil samples)
    n_samples = 60
    coords = np.random.rand(n_samples, 2) * 1000  # 1km x 1km study area

    # Create contamination plume (spatially correlated)
    # Source at (300, 300) with dispersion
    source_x, source_y = 300, 300
    distances = np.sqrt((coords[:, 0] - source_x) ** 2 + (coords[:, 1] - source_y) ** 2)

    # Contamination concentration (ppm) - decreases with distance from source
    concentration = (
        50 * np.exp(-distances / 200)  # Exponential decay
        + 5 * np.random.randn(n_samples)  # Random noise
    )
    concentration = np.maximum(0.1, concentration)  # Minimum detection limit

    # Regulatory threshold
    threshold = 10.0  # 10 ppm regulatory limit

    return coords, concentration, threshold


def main():
    """Run environmental risk assessment workflow."""
    print("=" * 80)
    print("ENVIRONMENTAL RISK ASSESSMENT WORKFLOW")
    print("=" * 80)
    print("\nThis demo shows how to assess environmental contamination risk using")
    print("spatial analysis and geostatistical methods.\n")

    # ============================================================================
    # STEP 1: Load Sample Data
    # ============================================================================
    print("STEP 1: Load Sample Data")
    print("-" * 80)

    coords, concentration, threshold = create_synthetic_contamination_data()
    points = PointSet(coordinates=coords)

    print(f"  Loaded {len(points.coordinates)} sample locations")
    print(f"  Contamination concentration statistics:")
    print(f"    Mean: {concentration.mean():.2f} ppm")
    print(f"    Std:  {concentration.std():.2f} ppm")
    print(f"    Min:  {concentration.min():.2f} ppm")
    print(f"    Max:  {concentration.max():.2f} ppm")
    print(f"    P95:  {np.percentile(concentration, 95):.2f} ppm")
    print(f"  Regulatory threshold: {threshold} ppm")
    print(f"  Samples exceeding threshold: {(concentration > threshold).sum()} ({100*(concentration > threshold).sum()/len(concentration):.1f}%)")

    # ============================================================================
    # STEP 2: Spatial Autocorrelation Analysis
    # ============================================================================
    print("\nSTEP 2: Spatial Autocorrelation Analysis")
    print("-" * 80)

    # Create spatial weights
    weights = create_knn_weights(points, k=8)

    # Test spatial autocorrelation
    moran_result = morans_i(concentration, weights)
    print(f"  Moran's I: {moran_result.I:.4f}")
    print(f"    Z-score: {moran_result.z_score:.2f}")
    print(f"    P-value: {moran_result.p_value:.4f}")

    if moran_result.p_value < 0.05:
        if moran_result.I > 0:
            print("    → Significant positive autocorrelation (contamination clusters)")
        else:
            print("    → Significant negative autocorrelation (contamination disperses)")
    else:
        print("    → No significant spatial autocorrelation")

    # ============================================================================
    # STEP 3: Hotspot Detection
    # ============================================================================
    print("\nSTEP 3: Hotspot Detection")
    print("-" * 80)

    hotspot_result = getis_ord_gi_star(concentration, weights)
    print(f"  Getis-Ord Gi* results:")
    print(f"    Significant hotspots: {hotspot_result.hotspots.sum()}")
    print(f"    Significant coldspots: {hotspot_result.coldspots.sum()}")

    if hotspot_result.hotspots.sum() > 0:
        hotspot_coords = coords[hotspot_result.hotspots]
        hotspot_conc = concentration[hotspot_result.hotspots]
        print(f"    Hotspot concentrations: {hotspot_conc.mean():.2f} ± {hotspot_conc.std():.2f} ppm")

    # ============================================================================
    # STEP 4: Contamination Mapping with Kriging
    # ============================================================================
    print("\nSTEP 4: Contamination Mapping with Kriging")
    print("-" * 80)

    # Compute variogram
    print("  Computing variogram...")
    lags, semi_vars, _ = compute_experimental_variogram(
        points, concentration, n_lags=15
    )

    # Fit variogram
    variogram = fit_variogram_model(lags, semi_vars, model_type="spherical")
    print(f"  Fitted {variogram.model_type} variogram:")
    print(f"    Range: {variogram.range_param:.0f} m")
    print(f"    R²: {variogram.r_squared:.4f}")

    # Create query grid
    grid_x = np.linspace(0, 1000, 40)
    grid_y = np.linspace(0, 1000, 40)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
    grid_coords = np.column_stack([grid_xx.ravel(), grid_yy.ravel()])
    grid_points = PointSet(coordinates=grid_coords)

    # Estimate concentration
    print("  Estimating concentration on grid...")
    model = GeostatisticalModel(
        data=points,
        values=concentration,
        method="kriging",
        kriging_type="ordinary",
        variogram_model=variogram,
    )

    conc_results = model.estimate(grid_points, return_variance=True)

    print(f"    Mean estimated concentration: {conc_results.estimates.mean():.2f} ppm")
    print(f"    Max estimated concentration: {conc_results.estimates.max():.2f} ppm")
    print(f"    Grid points above threshold: {(conc_results.estimates > threshold).sum()}")

    # ============================================================================
    # STEP 5: Risk Assessment with Indicator Kriging
    # ============================================================================
    print("\nSTEP 5: Risk Assessment with Indicator Kriging")
    print("-" * 80)
    print(f"  Computing probability of exceeding {threshold} ppm...")

    # Indicator kriging
    ik = IndicatorKriging(variogram_model=variogram, threshold=threshold)
    ik.fit(points, concentration)

    prob_result = ik.predict(grid_points)
    probabilities = prob_result.predictions

    print(f"    Mean probability: {probabilities.mean():.1%}")
    print(f"    High-risk areas (>70% probability): {(probabilities > 0.7).sum()} grid points")
    print(f"    Medium-risk areas (50-70% probability): {((probabilities >= 0.5) & (probabilities <= 0.7)).sum()} grid points")
    print(f"    Low-risk areas (<50% probability): {(probabilities < 0.5).sum()} grid points")

    # ============================================================================
    # STEP 6: Uncertainty Quantification with SGS
    # ============================================================================
    print("\nSTEP 6: Uncertainty Quantification with Sequential Gaussian Simulation")
    print("-" * 80)

    n_realizations = 50
    conc_realizations = sequential_gaussian_simulation(
        points,
        concentration,
        grid_points,
        variogram,
        n_realizations=n_realizations,
        random_seed=42,
    )

    # Exceedance probability from SGS
    prob_exceed_sgs = compute_exceedance_probability(conc_realizations, threshold)

    print(f"  Generated {n_realizations} realizations")
    print(f"  Exceedance probability (SGS):")
    print(f"    Mean: {prob_exceed_sgs.mean():.1%}")
    print(f"    High-risk areas (>70%): {(prob_exceed_sgs > 0.7).sum()}")

    # Uncertainty statistics
    conc_p10 = np.percentile(conc_realizations, 10, axis=0)
    conc_p50 = np.percentile(conc_realizations, 50, axis=0)
    conc_p90 = np.percentile(conc_realizations, 90, axis=0)

    print(f"  Concentration uncertainty:")
    print(f"    P10 (conservative): {conc_p10.mean():.2f} ppm")
    print(f"    P50 (median): {conc_p50.mean():.2f} ppm")
    print(f"    P90 (optimistic): {conc_p90.mean():.2f} ppm")

    # ============================================================================
    # STEP 7: Risk Classification
    # ============================================================================
    print("\nSTEP 7: Risk Classification")
    print("-" * 80)

    # Classify risk levels
    risk_levels = np.zeros(len(grid_points.coordinates), dtype=int)
    risk_levels[prob_exceed_sgs < 0.3] = 1  # Low risk
    risk_levels[(prob_exceed_sgs >= 0.3) & (prob_exceed_sgs < 0.7)] = 2  # Medium risk
    risk_levels[prob_exceed_sgs >= 0.7] = 3  # High risk

    print(f"  Risk classification:")
    print(f"    Low risk (<30%): {(risk_levels == 1).sum()} grid points")
    print(f"    Medium risk (30-70%): {(risk_levels == 2).sum()} grid points")
    print(f"    High risk (>70%): {(risk_levels == 3).sum()} grid points")

    # ============================================================================
    # STEP 8: Remediation Recommendations
    # ============================================================================
    print("\nSTEP 8: Remediation Recommendations")
    print("-" * 80)

    high_risk_areas = risk_levels == 3
    if high_risk_areas.sum() > 0:
        high_risk_coords = grid_coords[high_risk_areas]
        high_risk_conc = conc_results.estimates[high_risk_areas]

        print(f"  High-priority remediation areas: {high_risk_areas.sum()} locations")
        print(f"    Average concentration: {high_risk_conc.mean():.2f} ppm")
        print(f"    Max concentration: {high_risk_conc.max():.2f} ppm")
        print(f"    Average exceedance probability: {prob_exceed_sgs[high_risk_areas].mean():.1%}")

        # Estimate area (assuming 25m x 25m grid cells)
        cell_area = 25 * 25  # m²
        total_area = high_risk_areas.sum() * cell_area / 10000  # hectares
        print(f"    Estimated area: {total_area:.2f} hectares")

    # ============================================================================
    # STEP 9: Results Summary
    # ============================================================================
    print("\n" + "=" * 80)
    print("WORKFLOW SUMMARY")
    print("=" * 80)
    print(f"✓ Analyzed {len(points.coordinates)} sample locations")
    print(f"✓ Detected spatial autocorrelation (Moran's I = {moran_result.I:.4f})")
    print(f"✓ Identified {hotspot_result.hotspots.sum()} contamination hotspots")
    print(f"✓ Mapped contamination on {len(grid_points.coordinates)} grid points")
    print(f"✓ Assessed risk with indicator kriging and SGS")
    print(f"✓ Classified {high_risk_areas.sum()} high-risk areas for remediation")
    print("\nThis workflow demonstrates:")
    print("  • Spatial autocorrelation analysis")
    print("  • Hotspot detection")
    print("  • Contamination mapping with kriging")
    print("  • Risk assessment with indicator kriging")
    print("  • Uncertainty quantification with SGS")
    print("  • Remediation prioritization")
    print("=" * 80)

    # Create results DataFrame
    results_df = pd.DataFrame({
        "X": grid_coords[:, 0],
        "Y": grid_coords[:, 1],
        "Concentration": conc_results.estimates,
        "Concentration_Variance": conc_results.variance,
        "Concentration_P10": conc_p10,
        "Concentration_P50": conc_p50,
        "Concentration_P90": conc_p90,
        "ProbExceed_IK": probabilities,
        "ProbExceed_SGS": prob_exceed_sgs,
        "RiskLevel": risk_levels,
    })

    return results_df


if __name__ == "__main__":
    results = main()
    print(f"\nResults DataFrame shape: {results.shape}")
    print(f"Columns: {', '.join(results.columns)}")

