# GeoSmith Workflow Demos

This directory contains comprehensive workflow demonstrations that show complete,
production-ready geostatistical and spatial analysis workflows.

## Available Workflows

### 1. Ore Reserve Estimation (`ore_reserve_estimation.py`)

**Complete mining workflow** from drillhole data to block model estimation.

**Demonstrates:**
- Drillhole data processing
- Variogram analysis and model selection
- Cross-validation for model assessment
- Block model creation
- Grade estimation with kriging
- Uncertainty quantification with SGS
- Risk assessment with indicator kriging
- Grade-tonnage analysis
- GSLIB format export

**Use Cases:**
- Mine planning
- Ore reserve estimation
- Grade control
- Resource modeling

**Run:**
```bash
python examples/workflows/ore_reserve_estimation.py
```

---

### 2. Spatial Policy Analysis (`spatial_policy_analysis.py`)

**Public policy workflow** for location-based decision making.

**Demonstrates:**
- Spatial autocorrelation analysis (Moran's I)
- Hotspot detection (Getis-Ord Gi*)
- Multi-criteria decision analysis
- Priority area identification
- Policy recommendations

**Use Cases:**
- Environmental justice analysis
- Resource allocation decisions
- Infrastructure planning
- Public health interventions
- Fleet electrification prioritization (SDG&E case study)

**Run:**
```bash
python examples/workflows/spatial_policy_analysis.py
```

---

### 3. Reservoir Property Modeling (`reservoir_property_modeling.py`)

**Petroleum reservoir characterization workflow** from well logs to property maps.

**Demonstrates:**
- Well log data processing
- Petrophysical property calculation (porosity, permeability, water saturation)
- 3D spatial modeling with kriging
- Uncertainty quantification
- Multi-property reservoir characterization

**Use Cases:**
- Reservoir characterization
- Property mapping
- Well placement optimization
- Reserve estimation

**Run:**
```bash
python examples/workflows/reservoir_property_modeling.py
```

---

### 4. Environmental Risk Assessment (`environmental_risk_assessment.py`)

**Environmental contamination assessment workflow** with risk mapping.

**Demonstrates:**
- Sample data analysis
- Spatial autocorrelation
- Contamination mapping
- Risk assessment with indicator kriging
- Exceedance probability mapping
- Hotspot identification
- Remediation prioritization

**Use Cases:**
- Environmental site assessment
- Contamination monitoring
- Regulatory compliance
- Remediation planning

**Run:**
```bash
python examples/workflows/environmental_risk_assessment.py
```

---

## Workflow Structure

Each workflow follows a consistent structure:

1. **Data Loading** - Load and prepare input data
2. **Exploratory Analysis** - Understand data patterns
3. **Model Fitting** - Fit geostatistical models
4. **Validation** - Cross-validate model performance
5. **Estimation** - Generate predictions/estimates
6. **Uncertainty** - Quantify uncertainty
7. **Risk Assessment** - Assess risks and probabilities
8. **Results Export** - Export for further analysis

## Key Features Demonstrated

### Geostatistical Methods
- ✅ Variogram analysis and fitting
- ✅ Ordinary, Simple, Universal, and Indicator Kriging
- ✅ Sequential Gaussian Simulation (SGS)
- ✅ Cross-validation

### Spatial Analysis
- ✅ Spatial autocorrelation (Moran's I, Geary's C)
- ✅ Hotspot detection (Getis-Ord Gi*)
- ✅ Spatial weights generation

### Workflow Integration
- ✅ Unified `GeostatisticalModel` interface
- ✅ Complete data processing pipelines
- ✅ Industry-standard export formats (GSLIB)

## Running All Workflows

To run all workflow demos:

```bash
# Run all workflows
for workflow in examples/workflows/*.py; do
    if [ "$(basename $workflow)" != "__init__.py" ] && [ "$(basename $workflow)" != "README.md" ]; then
        echo "Running $(basename $workflow)..."
        python "$workflow"
        echo ""
    fi
done
```

## Customizing Workflows

Each workflow uses synthetic data for demonstration. To use your own data:

1. **Replace data loading functions** with your data sources
2. **Adjust parameters** (block sizes, variogram models, etc.)
3. **Modify thresholds** (cutoff grades, regulatory limits, etc.)
4. **Customize exports** for your software requirements

## Best Practices

These workflows demonstrate best practices:

- ✅ **Always cross-validate** models before use
- ✅ **Quantify uncertainty** with multiple realizations
- ✅ **Assess risk** with indicator kriging or exceedance probabilities
- ✅ **Export results** in industry-standard formats
- ✅ **Document assumptions** and parameters used

## Next Steps

After running these workflows:

1. **Adapt to your data** - Replace synthetic data with real data
2. **Tune parameters** - Adjust variogram models, block sizes, etc.
3. **Validate results** - Compare with known values or expert knowledge
4. **Iterate** - Refine models based on validation results

## References

- Isaaks, E.H. and Srivastava, R.M. (1989). *An Introduction to Applied Geostatistics*
- Goovaerts, P. (1997). *Geostatistics for Natural Resources Evaluation*
- Chiles, J.P. and Delfiner, P. (2012). *Geostatistics: Modeling Spatial Uncertainty*

