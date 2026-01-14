# Geostatistical Workflow Guide

This guide demonstrates complete geostatistical workflows using GeoSmith, from data preparation through estimation and validation.

## Table of Contents

1. [Complete Ore Reserve Estimation Workflow](#complete-ore-reserve-estimation-workflow)
2. [Variogram Analysis and Model Selection](#variogram-analysis-and-model-selection)
3. [Cross-Validation for Model Assessment](#cross-validation-for-model-assessment)
4. [Risk Assessment with Indicator Kriging](#risk-assessment-with-indicator-kriging)
5. [Handling Anisotropy and Nested Structures](#handling-anisotropy-and-nested-structures)
6. [Best Practices](#best-practices)

---

## Complete Ore Reserve Estimation Workflow

### Step 1: Load and Prepare Drillhole Data

```python
import numpy as np
import pandas as pd
from geosmith import PointSet
from geosmith.workflows.drillhole import (
    compute_3d_coordinates,
    merge_collar_assay,
    process_drillhole_data,
)

# Load collar and assay data
collar_df = pd.read_csv("collar_data.csv")
assay_df = pd.read_csv("assay_data.csv")

# Process and merge
column_map = process_drillhole_data(collar_df, assay_df)
merged_df = merge_collar_assay(collar_df, assay_df, column_map)

# Compute 3D coordinates
points = compute_3d_coordinates(merged_df, assume_vertical=True)
grades = merged_df["Au"].values  # Gold grades in g/t

print(f"Loaded {len(points.coordinates)} samples")
print(f"Grade statistics: mean={grades.mean():.2f} g/t, std={grades.std():.2f} g/t")
```

### Step 2: Exploratory Data Analysis

```python
from geosmith.primitives.variogram import compute_experimental_variogram
import matplotlib.pyplot as plt

# Compute experimental variogram
lags, semi_vars, n_pairs = compute_experimental_variogram(
    points, grades, n_lags=20
)

# Plot experimental variogram
plt.figure(figsize=(10, 6))
plt.scatter(lags, semi_vars, s=n_pairs, alpha=0.6, label="Experimental")
plt.xlabel("Lag Distance (m)")
plt.ylabel("Semi-Variance")
plt.title("Experimental Variogram")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Check for anisotropy by computing directional variograms
from geosmith.primitives.variogram_advanced import compute_directional_variogram

directions = [0, 45, 90, 135]  # East, NE, North, NW
for direction in directions:
    lags_dir, semi_vars_dir, _ = compute_directional_variogram(
        points, grades, direction=direction, angle_tolerance=22.5
    )
    plt.plot(lags_dir, semi_vars_dir, label=f"{direction}°")
plt.legend()
plt.show()
```

### Step 3: Fit Variogram Model

```python
from geosmith.primitives.variogram import fit_variogram_model

# Fit spherical model
variogram = fit_variogram_model(lags, semi_vars, model_type="spherical")

print(f"Fitted {variogram.model_type} model:")
print(f"  Nugget: {variogram.nugget:.4f}")
print(f"  Sill: {variogram.sill:.4f}")
print(f"  Range: {variogram.range_param:.2f} m")
print(f"  R²: {variogram.r_squared:.4f}")

# If nested structure is needed (common in mining)
from geosmith.primitives.variogram_advanced import fit_nested_variogram

nested_variogram = fit_nested_variogram(
    points, grades,
    n_components=3,
    component_types=["spherical", "spherical", "exponential"]
)
print(f"\nNested structure: {nested_variogram}")
```

### Step 4: Create Block Model Grid

```python
from geosmith.tasks.blockmodeltask import create_block_model_grid

# Create 3D block model grid
coords = points.coordinates
grid_coords, grid_info = create_block_model_grid(
    coords,
    block_size_xy=10.0,  # 10m x 10m blocks
    block_size_z=5.0,    # 5m vertical
    quantile_padding=0.05
)

print(f"Block model: {grid_info['nx']} x {grid_info['ny']} x {grid_info['nz']}")
print(f"Total blocks: {grid_info['n_blocks']}")

grid_points = PointSet(coordinates=grid_coords)
```

### Step 5: Estimate Grades with Cross-Validation

```python
from geosmith.workflows.geostatistics import GeostatisticalModel

# Create model with automatic validation
model = GeostatisticalModel(
    data=points,
    values=grades,
    method="kriging",
    kriging_type="universal",  # Accounts for trends
    validation="cross_validate",  # Leave-one-out CV
    drift_terms=["linear"],
    variogram_model=variogram
)

# Estimate on block model
results = model.estimate(grid_points, return_variance=True)

# Check validation metrics
if results.cv_result:
    print(f"\nCross-Validation Results:")
    print(f"  RMSE: {results.cv_result.rmse:.2f} g/t")
    print(f"  R²: {results.cv_result.r2:.3f}")
    print(f"  Mean Error (Bias): {results.cv_result.mean_error:.4f} g/t")
    print(f"  Std Error: {results.cv_result.std_error:.2f} g/t")
    
    # Good model should have:
    # - R² > 0.7
    # - Mean error close to 0 (unbiased)
    # - RMSE reasonable for application
```

### Step 6: Generate Uncertainty Realizations

```python
from geosmith.primitives.simulation import (
    sequential_gaussian_simulation,
    compute_exceedance_probability,
)

# Generate multiple realizations for uncertainty quantification
n_realizations = 100
realizations = sequential_gaussian_simulation(
    points, grades, grid_points, variogram,
    n_realizations=n_realizations,
    random_seed=42
)

# Compute exceedance probability (risk of exceeding cutoff)
cutoff = 2.0  # 2 g/t Au cutoff
prob_exceed = compute_exceedance_probability(realizations, cutoff)

print(f"\nUncertainty Analysis:")
print(f"  Mean grade: {results.estimates.mean():.2f} g/t")
print(f"  P10: {np.percentile(realizations, 10, axis=0).mean():.2f} g/t")
print(f"  P50: {np.percentile(realizations, 50, axis=0).mean():.2f} g/t")
print(f"  P90: {np.percentile(realizations, 90, axis=0).mean():.2f} g/t")
print(f"  Probability > {cutoff} g/t: {prob_exceed.mean():.1%}")
```

### Step 7: Export Results

```python
from geosmith.workflows.gslib import export_block_model_gslib

# Create block model DataFrame
block_model_df = pd.DataFrame({
    "X": grid_coords[:, 0],
    "Y": grid_coords[:, 1],
    "Z": grid_coords[:, 2],
    "Estimate": results.estimates,
    "Variance": results.variance,
    "ProbExceed": prob_exceed,
})

# Export to GSLIB format (industry standard)
export_block_model_gslib(
    block_model_df,
    "ore_reserve_model.dat",
    title="Ore Reserve Estimation - Gold Deposit"
)
```

---

## Variogram Analysis and Model Selection

### Comparing Different Variogram Models

```python
from geosmith.primitives.variogram import fit_variogram_model
from geosmith.primitives.kriging_cv import leave_one_out_cross_validation

# Compute experimental variogram
lags, semi_vars, _ = compute_experimental_variogram(points, grades, n_lags=20)

# Try different models
models = ["spherical", "exponential", "gaussian"]
results = {}

for model_type in models:
    # Fit model
    variogram = fit_variogram_model(lags, semi_vars, model_type=model_type)
    
    # Cross-validate
    cv = leave_one_out_cross_validation(
        points, grades, variogram, kriging_type="ordinary"
    )
    
    results[model_type] = {
        "variogram": variogram,
        "cv": cv
    }
    
    print(f"{model_type:12s}: R²={cv.r2:.3f}, RMSE={cv.rmse:.2f} g/t")

# Select best model
best_model = max(results, key=lambda k: results[k]["cv"].r2)
print(f"\nBest model: {best_model}")
```

### Handling Anisotropy

```python
from geosmith.primitives.variogram_advanced import (
    AnisotropicVariogramModel,
    compute_directional_variogram,
)

# Compute variograms in different directions
directions = [0, 45, 90, 135]
directional_variograms = {}

for direction in directions:
    lags_dir, semi_vars_dir, _ = compute_directional_variogram(
        points, grades, direction=direction
    )
    variogram_dir = fit_variogram_model(lags_dir, semi_vars_dir)
    directional_variograms[direction] = variogram_dir

# Check for anisotropy (different ranges in different directions)
ranges = {d: v.range_param for d, v in directional_variograms.items()}
anisotropy_ratio = max(ranges.values()) / min(ranges.values())

if anisotropy_ratio > 1.5:
    print(f"Anisotropy detected: ratio = {anisotropy_ratio:.2f}")
    
    # Fit anisotropic model
    base_variogram = fit_variogram_model(lags, semi_vars)
    major_range = max(ranges.values())
    anisotropy_angle = max(ranges, key=ranges.get)
    
    anisotropic_model = AnisotropicVariogramModel(
        base_model=base_variogram,
        anisotropy_ratio=anisotropy_ratio,
        anisotropy_angle=anisotropy_angle,
        major_range=major_range
    )
    print(f"Anisotropic model: {anisotropic_model}")
```

---

## Cross-Validation for Model Assessment

### Comprehensive Model Validation

```python
from geosmith.primitives.kriging_cv import (
    leave_one_out_cross_validation,
    k_fold_cross_validation,
)

# Leave-one-out CV (most thorough but slow)
cv_loo = leave_one_out_cross_validation(
    points, grades, variogram, kriging_type="ordinary"
)

# K-fold CV (faster for large datasets)
cv_kfold = k_fold_cross_validation(
    points, grades, variogram,
    n_folds=5,
    kriging_type="ordinary"
)

print("Leave-One-Out CV:")
print(f"  RMSE: {cv_loo.rmse:.2f} g/t")
print(f"  R²: {cv_loo.r2:.3f}")
print(f"  Bias: {cv_loo.mean_error:.4f} g/t")

print("\nK-Fold CV:")
print(f"  RMSE: {cv_kfold.rmse:.2f} g/t")
print(f"  R²: {cv_kfold.r2:.3f}")
print(f"  Bias: {cv_kfold.mean_error:.4f} g/t")

# Plot prediction vs. observed
plt.figure(figsize=(8, 8))
plt.scatter(grades, cv_loo.predictions, alpha=0.6)
plt.plot([grades.min(), grades.max()], [grades.min(), grades.max()], 'r--')
plt.xlabel("Observed Grade (g/t)")
plt.ylabel("Predicted Grade (g/t)")
plt.title("Cross-Validation: Predicted vs. Observed")
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Risk Assessment with Indicator Kriging

### Probability Maps for Economic Thresholds

```python
from geosmith.primitives.kriging import IndicatorKriging

# Define economic threshold
cutoff = 2.0  # 2 g/t Au

# Fit indicator kriging
ik = IndicatorKriging(variogram_model=variogram, threshold=cutoff)
ik.fit(points, grades)

# Predict probability of exceeding threshold
prob_result = ik.predict(grid_points)
probabilities = prob_result.predictions  # Values in [0, 1]

# Identify high-probability areas
high_prob = probabilities > 0.7
print(f"Blocks with >70% probability of exceeding {cutoff} g/t: {high_prob.sum()}")

# Create risk map
risk_categories = np.zeros_like(probabilities, dtype=int)
risk_categories[probabilities < 0.3] = 1  # Low risk
risk_categories[(probabilities >= 0.3) & (probabilities < 0.7)] = 2  # Medium risk
risk_categories[probabilities >= 0.7] = 3  # High risk
```

---

## Best Practices

### 1. Data Quality Checks

```python
# Check for outliers
from scipy import stats

z_scores = np.abs(stats.zscore(grades))
outliers = z_scores > 3
print(f"Potential outliers: {outliers.sum()}")

# Check for duplicate coordinates
from scipy.spatial.distance import cdist
distances = cdist(points.coordinates, points.coordinates)
duplicates = (distances < 1e-6).sum() - len(points.coordinates)
print(f"Duplicate locations: {duplicates}")
```

### 2. Variogram Fitting Guidelines

- **Use at least 15-20 lag bins** for reliable variogram estimation
- **Check directional variograms** to detect anisotropy
- **Compare multiple models** using cross-validation
- **Consider nested structures** if variogram shows multiple scales
- **Validate with cross-validation** - R² > 0.7 is good

### 3. Kriging Type Selection

- **Ordinary Kriging**: Default choice, assumes unknown constant mean
- **Simple Kriging**: Use when mean is known from prior knowledge
- **Universal Kriging**: Use when clear spatial trends exist
- **Indicator Kriging**: Use for risk assessment or categorical variables

### 4. Performance Optimization

```python
# For large datasets, use k-fold instead of leave-one-out
cv = k_fold_cross_validation(
    points, grades, variogram,
    n_folds=5,  # Faster than leave-one-out
    kriging_type="ordinary"
)

# Use Numba-accelerated variogram computation (automatic)
# For very large grids, consider sub-sampling for initial exploration
```

### 5. Uncertainty Quantification

- **Always generate multiple realizations** (50-100 minimum)
- **Compute exceedance probabilities** for risk assessment
- **Report P10/P50/P90** statistics, not just mean
- **Visualize uncertainty** with probability maps

---

## Common Pitfalls and Solutions

### Problem: Poor Cross-Validation Results

**Symptoms**: Low R², high RMSE, large bias

**Solutions**:
1. Check for outliers and data quality issues
2. Try different variogram models
3. Consider nested structures or anisotropy
4. Use Universal Kriging if trends exist
5. Increase number of samples if possible

### Problem: Anisotropy Not Detected

**Solutions**:
1. Compute directional variograms in multiple directions
2. Use angle tolerance of 22.5° or less
3. Ensure sufficient data in each direction
4. Consider zonal anisotropy if geometric anisotropy doesn't fit

### Problem: Nested Structure Needed

**Symptoms**: Experimental variogram shows multiple "steps"

**Solutions**:
1. Use `fit_nested_variogram()` with 2-3 components
2. Start with same model type for all components
3. Check that ranges are well-separated
4. Validate with cross-validation

---

## References

- Isaaks, E.H. and Srivastava, R.M. (1989). *An Introduction to Applied Geostatistics*
- Goovaerts, P. (1997). *Geostatistics for Natural Resources Evaluation*
- Chiles, J.P. and Delfiner, P. (2012). *Geostatistics: Modeling Spatial Uncertainty*

