"""Geomechanics: Field-wide analysis

Pure geomechanics operations - field module.
Migrated from geosuite.geomech.
Layer 2: Primitives - Pure operations.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

from geosmith.primitives.geomechanics._common import (
    NUMBA_AVAILABLE,
    PANDAS_AVAILABLE,
    njit,
    pd,
)

if TYPE_CHECKING:
    import pandas as pd

from dataclasses import dataclass


@dataclass
class FieldOptimizationResult:
    """Results from field-wide drilling optimization.

    Attributes:
        optimal_mud_weights: Dictionary mapping well name to optimal mud weight (g/cc).
        cost_savings: Total cost savings (currency units).
        time_savings: Total time savings (days).
        risk_reduction: Risk reduction factor (0-1).
        success_probability: Field success probability (0-1).
        recommendations: List of optimization recommendations.
    """

    optimal_mud_weights: Dict[str, float]
    cost_savings: float
    time_savings: float
    risk_reduction: float
    success_probability: float
    recommendations: List[str]

def process_field_data(df: "pd.DataFrame") -> Dict[str, "pd.DataFrame"]:
    """Process multi-well field data into separate well datasets.

    Args:
        df: Combined field data with 'well_name' column.

    Returns:
        Dictionary mapping well_name to well DataFrame.

    Example:
        >>> from geosmith.primitives.geomechanics import process_field_data
        >>> import pandas as pd
        >>>
        >>> df = pd.DataFrame({'well_name': ['Well1', 'Well1', 'Well2'], 'depth_m': [1000, 2000, 1500]})
        >>> wells = process_field_data(df)
        >>> print(f"Processed {len(wells)} wells")
    """
    if not PANDAS_AVAILABLE:
        raise ImportError(
            "pandas is required for process_field_data. "
            "Install with: pip install pandas"
        )

    wells = {}
    for well_name in df["well_name"].unique():
        well_df = df[df["well_name"] == well_name].copy()
        well_df = well_df.reset_index(drop=True)
        wells[well_name] = well_df
    return wells

def calculate_field_statistics(
    field_data: "pd.DataFrame",
) -> Dict[str, Any]:
    """Calculate field-wide drilling statistics and performance metrics.

    Args:
        field_data: Multi-well field data with columns:
            - well_name: Well identifier
            - mud_weight_used: Mud weight used (g/cc)
            - cost_per_meter: Drilling cost per meter
            - days_to_drill: Days to drill well
            - drilling_status: Drilling status ('Success' or failure type)
            - depth_m: Depth (m)
            - formation: Formation name (optional)
            - GR, RHOB, RT: Log measurements (optional)

    Returns:
        Dictionary with field statistics:
        - total_wells: Total number of wells
        - overall_success_rate: Overall success rate (0-1)
        - avg_cost_per_meter: Average cost per meter
        - avg_drilling_days: Average drilling days
        - well_statistics: DataFrame with per-well statistics
        - problem_types: Dictionary of problem type counts
        - formation_performance: DataFrame with formation-level statistics

    Example:
        >>> from geosmith.primitives.geomechanics import calculate_field_statistics
        >>> import pandas as pd
        >>>
        >>> field_data = pd.DataFrame({
        ...     'well_name': ['Well1', 'Well2'],
        ...     'mud_weight_used': [1.2, 1.3],
        ...     'cost_per_meter': [100, 120],
        ...     'days_to_drill': [10, 12],
        ...     'drilling_status': ['Success', 'Success'],
        ...     'depth_m': [2000, 2500]
        ... })
        >>> stats = calculate_field_statistics(field_data)
        >>> print(f"Success rate: {stats['overall_success_rate']:.1%}")
    """
    if not PANDAS_AVAILABLE:
        raise ImportError(
            "pandas is required for calculate_field_statistics. "
            "Install with: pip install pandas"
        )

    # Group by well for well-level statistics
    well_stats = field_data.groupby("well_name").agg(
        {
            "mud_weight_used": ["mean", "std"],
            "cost_per_meter": "first",
            "days_to_drill": "first",
            "drilling_status": lambda x: (x == "Success").mean(),
            "depth_m": ["min", "max"],
        }
    ).round(3)

    # Flatten column names
    well_stats.columns = [
        "_".join(col).strip() for col in well_stats.columns.values
    ]

    # Overall field statistics
    total_wells = field_data["well_name"].nunique()
    overall_success_rate = (
        field_data.groupby("well_name")["drilling_status"]
        .apply(lambda x: (x == "Success").all())
        .mean()
    )

    avg_cost_per_meter = (
        field_data.groupby("well_name")["cost_per_meter"].first().mean()
    )
    avg_drilling_days = (
        field_data.groupby("well_name")["days_to_drill"].first().mean()
    )

    # Problem analysis
    problems = field_data[field_data["drilling_status"] != "Success"]
    problem_types = problems["drilling_status"].value_counts().to_dict()

    # Formation analysis (if formation column exists)
    formation_performance = None
    if "formation" in field_data.columns:
        formation_performance = (
            field_data.groupby("formation")
            .agg(
                {
                    "drilling_status": lambda x: (x == "Success").mean(),
                    "mud_weight_used": "mean",
                    "GR": "mean" if "GR" in field_data.columns else "count",
                    "RHOB": "mean" if "RHOB" in field_data.columns else "count",
                    "RT": "mean" if "RT" in field_data.columns else "count",
                }
            )
            .round(3)
        )

    return {
        "total_wells": total_wells,
        "overall_success_rate": overall_success_rate,
        "avg_cost_per_meter": avg_cost_per_meter,
        "avg_drilling_days": avg_drilling_days,
        "well_statistics": well_stats,
        "problem_types": problem_types,
        "formation_performance": formation_performance,
    }

