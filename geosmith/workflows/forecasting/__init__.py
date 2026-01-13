"""
Forecasting and decline curve analysis module.

Provides tools for production forecasting including:
- Physics-informed decline models
- Bayesian posterior sampling
- Time series decomposition
- Scenario forecasting with economic inputs
- Monte Carlo ensembles
"""

__all__ = []

try:
    from geosmith.workflows.forecasting.decline_models import (
        DeclineModel,
        ExponentialDecline,
        HarmonicDecline,
        HyperbolicDecline,
        fit_decline_model,
        forecast_production,
        process_wells_parallel,
    )

    __all__.extend(
        [
            "DeclineModel",
            "ExponentialDecline",
            "HyperbolicDecline",
            "HarmonicDecline",
            "fit_decline_model",
            "forecast_production",
            "process_wells_parallel",
        ]
    )
except ImportError:
    pass

try:
    from geosmith.workflows.forecasting.bayesian_decline import (
        BayesianDeclineAnalyzer,
        forecast_with_uncertainty,
        sample_decline_posterior,
    )

    __all__.extend(
        [
            "BayesianDeclineAnalyzer",
            "sample_decline_posterior",
            "forecast_with_uncertainty",
        ]
    )
except ImportError:
    pass

try:
    from .decomposition import (
        decompose_production,
        detect_seasonality,
        detect_trend,
        remove_trend_seasonality,
    )

    __all__.extend(
        [
            "decompose_production",
            "detect_trend",
            "detect_seasonality",
            "remove_trend_seasonality",
        ]
    )
except ImportError:
    pass

try:
    from .scenario_forecasting import (
        ScenarioForecaster,
        create_scenarios,
        forecast_with_economics,
    )

    __all__.extend(
        ["ScenarioForecaster", "forecast_with_economics", "create_scenarios"]
    )
except ImportError:
    pass

try:
    from geosmith.workflows.forecasting.monte_carlo_forecast import (
        MonteCarloForecaster,
        ensemble_forecast,
        forecast_uncertainty_bands,
    )

    __all__.extend(
        ["MonteCarloForecaster", "ensemble_forecast", "forecast_uncertainty_bands"]
    )
except ImportError:
    pass

try:
    from geosmith.workflows.forecasting.production_analysis import (
        aggregate_by_county,
        aggregate_by_field,
        aggregate_by_pool,
        analyze_spatial_distribution,
        analyze_temporal_coverage,
        calculate_production_density,
        calculate_production_summary,
        calculate_well_statistics,
        identify_production_hotspots,
    )

    __all__.extend(
        [
            "analyze_temporal_coverage",
            "analyze_spatial_distribution",
            "aggregate_by_field",
            "aggregate_by_pool",
            "aggregate_by_county",
            "calculate_well_statistics",
            "calculate_production_summary",
            "calculate_production_density",
            "identify_production_hotspots",
        ]
    )
except ImportError:
    pass

try:
    from .validation import (
        calculate_forecast_metrics,
        calculate_summary_statistics,
        cross_validate_well,
        evaluate_wells_dataset,
        print_summary_statistics,
    )

    __all__.extend(
        [
            "calculate_forecast_metrics",
            "cross_validate_well",
            "evaluate_wells_dataset",
            "calculate_summary_statistics",
            "print_summary_statistics",
        ]
    )
except ImportError:
    pass
