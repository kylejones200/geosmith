"""Spatial Policy Analysis Workflow Demo.

This demo shows how to use spatial analysis tools for public policy evaluation,
demonstrating the SDG&E fleet electrification case study approach:

1. Load and prepare spatial data
2. Spatial autocorrelation analysis (Moran's I)
3. Hotspot detection (Getis-Ord Gi*)
4. Priority area identification
5. Policy recommendations

This workflow is useful for:
- Environmental justice analysis
- Resource allocation decisions
- Infrastructure planning
- Public health interventions
"""

import numpy as np
import pandas as pd

from geosmith import PointSet
from geosmith.primitives.spatial_analysis import (
    create_knn_weights,
    getis_ord_gi_star,
    morans_i,
)


def create_synthetic_policy_data():
    """Create synthetic data simulating SDG&E case study.

    In practice, this would be loaded from census, EPA, and health department data.
    """
    np.random.seed(42)

    # Create census tract centroids (simplified - in practice would use actual polygons)
    n_tracts = 150
    coords = np.random.rand(n_tracts, 2) * 1000  # 1000m x 1000m study area

    # Create spatially correlated variables
    # Population density (higher in some areas)
    pop_density = (
        5000
        + 3000 * np.exp(-((coords[:, 0] - 300) ** 2 + (coords[:, 1] - 300) ** 2) / 20000)
        + 2000 * np.exp(-((coords[:, 0] - 700) ** 2 + (coords[:, 1] - 700) ** 2) / 20000)
        + np.random.randn(n_tracts) * 500
    )
    pop_density = np.maximum(100, pop_density)  # Minimum density

    # Median income (inverse correlation with density in some areas)
    median_income = (
        60000
        - 20000 * (pop_density / pop_density.max())
        + np.random.randn(n_tracts) * 10000
    )
    median_income = np.maximum(20000, median_income)

    # Asthma rates (higher in high-density, low-income areas)
    asthma_rate = (
        0.05
        + 0.03 * (pop_density / pop_density.max())
        - 0.02 * (median_income / median_income.max())
        + np.random.randn(n_tracts) * 0.01
    )
    asthma_rate = np.maximum(0.01, np.minimum(0.15, asthma_rate))

    # Pollution levels (PM2.5) - higher near industrial areas
    pollution = (
        10
        + 5 * np.exp(-((coords[:, 0] - 200) ** 2 + (coords[:, 1] - 200) ** 2) / 15000)
        + np.random.randn(n_tracts) * 2
    )
    pollution = np.maximum(5, pollution)

    return coords, pop_density, median_income, asthma_rate, pollution


def main():
    """Run spatial policy analysis workflow."""
    print("=" * 80)
    print("SPATIAL POLICY ANALYSIS WORKFLOW")
    print("=" * 80)
    print("\nThis demo shows how to use spatial analysis for public policy,")
    print("demonstrating the SDG&E fleet electrification prioritization approach.\n")

    # ============================================================================
    # STEP 1: Data Loading and Preparation
    # ============================================================================
    print("STEP 1: Data Loading and Preparation")
    print("-" * 80)
    print("Loading census tract and environmental data...")

    coords, pop_density, median_income, asthma_rate, pollution = (
        create_synthetic_policy_data()
    )

    points = PointSet(coordinates=coords)

    print(f"  Loaded {len(points.coordinates)} census tracts")
    print(f"  Variables:")
    print(f"    Population density: {pop_density.mean():.0f} ± {pop_density.std():.0f} per km²")
    print(f"    Median income: ${median_income.mean():,.0f} ± ${median_income.std():,.0f}")
    print(f"    Asthma rate: {asthma_rate.mean():.1%} ± {asthma_rate.std():.1%}")
    print(f"    Pollution (PM2.5): {pollution.mean():.1f} ± {pollution.std():.1f} μg/m³")

    # ============================================================================
    # STEP 2: Spatial Autocorrelation Analysis
    # ============================================================================
    print("\nSTEP 2: Spatial Autocorrelation Analysis")
    print("-" * 80)
    print("Creating spatial weights matrix...")

    # Create K-nearest neighbors weights (k=8 for census tracts)
    weights = create_knn_weights(points, k=8)
    print(f"  Spatial weights: {weights}")

    # Test spatial autocorrelation for each variable
    variables = {
        "Population Density": pop_density,
        "Median Income": median_income,
        "Asthma Rate": asthma_rate,
        "Pollution (PM2.5)": pollution,
    }

    print("\n  Computing Moran's I for each variable:")
    print(f"  {'Variable':<20} {'Moran\'s I':>12} {'Z-score':>10} {'P-value':>10} {'Significance':>12}")
    print("  " + "-" * 64)

    moran_results = {}
    for var_name, values in variables.items():
        result = morans_i(values, weights)
        moran_results[var_name] = result

        significance = ""
        if result.p_value < 0.001:
            significance = "***"
        elif result.p_value < 0.01:
            significance = "**"
        elif result.p_value < 0.05:
            significance = "*"

        print(
            f"  {var_name:<20} {result.I:>10.4f} {result.z_score:>8.2f} "
            f"{result.p_value:>8.4f} {significance:>10}"
        )

    # ============================================================================
    # STEP 3: Hotspot Detection
    # ============================================================================
    print("\nSTEP 3: Hotspot Detection (Getis-Ord Gi*)")
    print("-" * 80)
    print("Identifying statistically significant hotspots and coldspots...")

    hotspot_results = {}
    for var_name, values in variables.items():
        result = getis_ord_gi_star(values, weights)
        hotspot_results[var_name] = result

        print(f"\n  {var_name}:")
        print(f"    Hotspots:  {result.hotspots.sum()} tracts")
        print(f"    Coldspots: {result.coldspots.sum()} tracts")
        print(f"    Mean Gi*:  {result.gi_star.mean():.4f}")

    # ============================================================================
    # STEP 4: Priority Score Calculation
    # ============================================================================
    print("\nSTEP 4: Priority Score Calculation")
    print("-" * 80)
    print("Computing priority scores for fleet electrification...")
    print("  Priority = f(population density, income, asthma, pollution)")

    # Normalize variables to [0, 1] scale
    pop_norm = (pop_density - pop_density.min()) / (pop_density.max() - pop_density.min())
    income_norm = 1 - (median_income - median_income.min()) / (
        median_income.max() - median_income.min()
    )  # Inverse (lower income = higher priority)
    asthma_norm = (asthma_rate - asthma_rate.min()) / (asthma_rate.max() - asthma_rate.min())
    pollution_norm = (pollution - pollution.min()) / (pollution.max() - pollution.min())

    # Compute priority score (weighted combination)
    priority_score = (
        0.3 * pop_norm
        + 0.2 * income_norm
        + 0.25 * asthma_norm
        + 0.25 * pollution_norm
    )

    print(f"  Priority score statistics:")
    print(f"    Mean: {priority_score.mean():.3f}")
    print(f"    Std:  {priority_score.std():.3f}")
    print(f"    Min:  {priority_score.min():.3f}")
    print(f"    Max:  {priority_score.max():.3f}")

    # Identify high-priority areas (top 20%)
    threshold = np.percentile(priority_score, 80)
    high_priority = priority_score >= threshold

    print(f"\n  High-priority areas (top 20%, score >= {threshold:.3f}):")
    print(f"    Number of tracts: {high_priority.sum()}")
    print(f"    Average population density: {pop_density[high_priority].mean():.0f} per km²")
    print(f"    Average median income: ${median_income[high_priority].mean():,.0f}")
    print(f"    Average asthma rate: {asthma_rate[high_priority].mean():.1%}")
    print(f"    Average pollution: {pollution[high_priority].mean():.1f} μg/m³")

    # ============================================================================
    # STEP 5: Spatial Analysis of Priority Scores
    # ============================================================================
    print("\nSTEP 5: Spatial Analysis of Priority Scores")
    print("-" * 80)

    # Test spatial autocorrelation of priority scores
    priority_moran = morans_i(priority_score, weights)
    print(f"  Moran's I for priority scores: {priority_moran.I:.4f}")
    print(f"    Z-score: {priority_moran.z_score:.2f}")
    print(f"    P-value: {priority_moran.p_value:.4f}")

    if priority_moran.p_value < 0.05:
        if priority_moran.I > 0:
            print("    → Significant positive autocorrelation (priority areas cluster)")
        else:
            print("    → Significant negative autocorrelation (priority areas disperse)")

    # Hotspot detection on priority scores
    priority_hotspots = getis_ord_gi_star(priority_score, weights)
    print(f"\n  Priority score hotspots:")
    print(f"    Significant hotspots: {priority_hotspots.hotspots.sum()} tracts")
    print(f"    Significant coldspots: {priority_hotspots.coldspots.sum()} tracts")

    # ============================================================================
    # STEP 6: Policy Recommendations
    # ============================================================================
    print("\nSTEP 6: Policy Recommendations")
    print("-" * 80)

    # Identify priority zones
    priority_zones = high_priority & priority_hotspots.hotspots
    n_zones = priority_zones.sum()

    print(f"  Recommended priority zones for fleet electrification: {n_zones} tracts")
    print(f"\n  Characteristics of priority zones:")

    if n_zones > 0:
        print(f"    Population density: {pop_density[priority_zones].mean():.0f} per km²")
        print(f"    Median income: ${median_income[priority_zones].mean():,.0f}")
        print(f"    Asthma rate: {asthma_rate[priority_zones].mean():.1%}")
        print(f"    Pollution level: {pollution[priority_zones].mean():.1f} μg/m³")
        print(f"    Average priority score: {priority_score[priority_zones].mean():.3f}")

        # Estimate impact
        total_pop_impact = pop_density[priority_zones].sum() / 1000  # in thousands
        print(f"\n  Estimated impact:")
        print(f"    Population affected: ~{total_pop_impact:.0f},000 people")
        print(f"    Potential pollution reduction: {pollution[priority_zones].mean() - pollution.mean():.1f} μg/m³")

    # ============================================================================
    # STEP 7: Results Summary
    # ============================================================================
    print("\n" + "=" * 80)
    print("WORKFLOW SUMMARY")
    print("=" * 80)
    print(f"✓ Analyzed {len(points.coordinates)} census tracts")
    print(f"✓ Tested spatial autocorrelation for {len(variables)} variables")
    print(f"✓ Identified hotspots and coldspots for each variable")
    print(f"✓ Computed priority scores for {high_priority.sum()} high-priority tracts")
    print(f"✓ Recommended {n_zones} priority zones for intervention")
    print("\nThis workflow demonstrates:")
    print("  • Spatial autocorrelation analysis (Moran's I)")
    print("  • Hotspot detection (Getis-Ord Gi*)")
    print("  • Multi-criteria decision analysis")
    print("  • Policy prioritization based on spatial patterns")
    print("=" * 80)

    # Return results for further analysis
    results_df = pd.DataFrame({
        "X": coords[:, 0],
        "Y": coords[:, 1],
        "PopDensity": pop_density,
        "MedianIncome": median_income,
        "AsthmaRate": asthma_rate,
        "Pollution": pollution,
        "PriorityScore": priority_score,
        "HighPriority": high_priority,
        "Hotspot": priority_hotspots.hotspots,
        "Coldspot": priority_hotspots.coldspots,
        "PriorityZone": priority_zones,
    })

    return results_df


if __name__ == "__main__":
    results = main()
    print(f"\nResults DataFrame shape: {results.shape}")
    print(f"Columns: {', '.join(results.columns)}")

