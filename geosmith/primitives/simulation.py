"""Sequential Gaussian Simulation (SGS) primitives.

Pure simulation operations that work with Layer 1 objects.
Migrated from geosuite.mining.geostatistics.
Layer 2: Primitives - Pure operations.
"""

import logging
from typing import Optional

import numpy as np

from geosmith.objects.pointset import PointSet
from geosmith.primitives.kriging import OrdinaryKriging
from geosmith.primitives.variogram import VariogramModel

logger = logging.getLogger(__name__)


def sequential_gaussian_simulation(
    samples: PointSet,
    sample_values: np.ndarray,
    query_points: PointSet,
    variogram_model: VariogramModel,
    n_realizations: int = 1,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """Perform Sequential Gaussian Simulation (SGS) for uncertainty quantification.

    SGS generates multiple realizations of a spatial field that honor:
    - Sample data values
    - Spatial correlation structure (variogram)
    - Statistical distribution

    Args:
        samples: PointSet with sample locations.
        sample_values: Sample values (n_samples,).
        query_points: PointSet with locations to simulate.
        variogram_model: Fitted variogram model.
        n_realizations: Number of realizations to generate.
        random_seed: Random seed for reproducibility.

    Returns:
        Array of shape (n_realizations, n_query_points) with simulated values.

    Raises:
        ValueError: If inputs are invalid.

    Example:
        >>> from geosmith import PointSet
        >>> from geosmith.primitives.simulation import sequential_gaussian_simulation
        >>> from geosmith.primitives.variogram import fit_variogram_model
        >>>
        >>> # Fit variogram
        >>> variogram = fit_variogram_model(lags, semi_vars)
        >>>
        >>> # Generate realizations
        >>> realizations = sequential_gaussian_simulation(
        ...     samples, sample_values, query_points, variogram,
        ...     n_realizations=100, random_seed=42
        ... )
        >>> print(f"Generated {realizations.shape[0]} realizations")
    """
    if len(samples.coordinates) != len(sample_values):
        raise ValueError(
            f"Samples ({len(samples.coordinates)}) and values ({len(sample_values)}) "
            f"must have same length"
        )

    if len(sample_values) < 3:
        raise ValueError(
            f"Need at least 3 samples for simulation, got {len(sample_values)}"
        )

    if n_realizations < 1:
        raise ValueError(f"n_realizations must be >= 1, got {n_realizations}")

    n_query = len(query_points.coordinates)
    logger.info(
        f"Starting SGS: {len(sample_values)} samples, "
        f"{n_query} query points, {n_realizations} realizations"
    )

    # Set random seed
    if random_seed is not None:
        np.random.seed(random_seed)

    # Check if data needs transformation (log-normal)
    # Simple heuristic: if values are all positive and skewed, suggest log transform
    if np.all(sample_values > 0) and np.std(sample_values) > np.mean(sample_values):
        logger.warning(
            "Data appears log-normally distributed. "
            "Consider log-transforming before simulation."
        )

    # Initialize output array
    realizations = np.zeros((n_realizations, n_query))

    # Create kriging model
    kriging = OrdinaryKriging(variogram_model=variogram_model)
    kriging.fit(samples, sample_values)

    # Generate each realization
    for i in range(n_realizations):
        logger.debug(f"Generating realization {i+1}/{n_realizations}")

        # Random path through query points
        path = np.random.permutation(n_query)
        query_path = PointSet(coordinates=query_points.coordinates[path])

        # Simulated values for this realization
        sim_values = np.zeros(n_query)

        # Sequential simulation
        for j, idx in enumerate(path):
            # Get current query point
            current_point = PointSet(
                coordinates=query_points.coordinates[idx : idx + 1]
            )

            # Combine samples with previously simulated points
            if j == 0:
                # First point: use only samples
                conditioning_points = samples
                conditioning_values = sample_values
            else:
                # Subsequent points: use samples + previously simulated
                simulated_coords = query_points.coordinates[path[:j]]
                simulated_vals = sim_values[path[:j]]

                # Combine
                all_coords = np.vstack([samples.coordinates, simulated_coords])
                all_values = np.hstack([sample_values, simulated_vals])
                conditioning_points = PointSet(coordinates=all_coords)
                conditioning_values = all_values

            # Kriging prediction
            kriging_fit = OrdinaryKriging(variogram_model=variogram_model)
            kriging_fit.fit(conditioning_points, conditioning_values)
            result = kriging_fit.predict(current_point, return_variance=True)

            # Draw from conditional distribution
            mean = result.predictions[0]
            std = np.sqrt(result.variance[0])

            # Ensure non-negative variance
            if std < 0:
                std = 0.0

            # Sample from normal distribution
            sim_values[idx] = np.random.normal(mean, std)

        realizations[i, :] = sim_values

    logger.info(f"Completed SGS: {n_realizations} realizations generated")

    return realizations


def compute_exceedance_probability(
    realizations: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Compute exceedance probability from SGS realizations.

    Exceedance probability = P(Z > threshold) = fraction of realizations above threshold.

    Args:
        realizations: Array of shape (n_realizations, n_points) from SGS.
        threshold: Threshold value.

    Returns:
        Array of shape (n_points,) with exceedance probabilities [0, 1].

    Example:
        >>> realizations = sequential_gaussian_simulation(...)
        >>> prob = compute_exceedance_probability(realizations, threshold=2.0)
        >>> print(f"Probability of grade > 2.0: {prob.mean():.2%}")
    """
    if realizations.ndim != 2:
        raise ValueError(
            f"realizations must be 2D array (n_realizations, n_points), "
            f"got shape {realizations.shape}"
        )

    # Count realizations above threshold for each point
    above_threshold = np.sum(realizations > threshold, axis=0)

    # Compute probability
    n_realizations = realizations.shape[0]
    probability = above_threshold / n_realizations

    return probability


def compute_simulation_statistics(
    realizations: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute statistics from SGS realizations.

    Args:
        realizations: Array of shape (n_realizations, n_points) from SGS.

    Returns:
        Dictionary with:
            - 'mean': Mean across realizations
            - 'std': Standard deviation across realizations
            - 'p10': 10th percentile
            - 'p50': 50th percentile (median)
            - 'p90': 90th percentile
            - 'min': Minimum across realizations
            - 'max': Maximum across realizations

    Example:
        >>> realizations = sequential_gaussian_simulation(...)
        >>> stats = compute_simulation_statistics(realizations)
        >>> print(f"Mean: {stats['mean'].mean():.2f}")
        >>> print(f"P90: {stats['p90'].mean():.2f}")
    """
    if realizations.ndim != 2:
        raise ValueError(
            f"realizations must be 2D array (n_realizations, n_points), "
            f"got shape {realizations.shape}"
        )

    return {
        "mean": np.mean(realizations, axis=0),
        "std": np.std(realizations, axis=0),
        "p10": np.percentile(realizations, 10, axis=0),
        "p50": np.percentile(realizations, 50, axis=0),
        "p90": np.percentile(realizations, 90, axis=0),
        "min": np.min(realizations, axis=0),
        "max": np.max(realizations, axis=0),
    }



def log_transform(values: np.ndarray, add_one: bool = True) -> np.ndarray:
    """Log transform values (optionally adding 1 to avoid log(0)).

    Useful for normalizing geostatistical data before simulation.

    Args:
        values: Values to transform.
        add_one: If True, use log1p (log(1+x)); if False, use log(x), default True.

    Returns:
        Log-transformed values.

    Example:
        >>> from geosmith.primitives.simulation import log_transform
        >>>
        >>> log_grades = log_transform(grades, add_one=True)
        >>> print(f"Transformed {len(grades)} values")
    """
    values = np.asarray(values, dtype=np.float64)

    if add_one:
        return np.log1p(values)
    else:
        return np.log(np.maximum(values, 1e-10))  # Avoid log(0)


def exp_transform(log_values: np.ndarray, subtract_one: bool = True) -> np.ndarray:
    """Exponential transform values back from log space.

    Inverse of log_transform. Useful for back-transforming simulated values.

    Args:
        log_values: Log-transformed values.
        subtract_one: If True, use expm1 (exp(x)-1); if False, use exp(x), default True.

    Returns:
        Original-scale values.

    Example:
        >>> from geosmith.primitives.simulation import exp_transform
        >>>
        >>> grades = exp_transform(log_grades, subtract_one=True)
        >>> print(f"Back-transformed {len(grades)} values")
    """
    log_values = np.asarray(log_values, dtype=np.float64)

    if subtract_one:
        return np.expm1(log_values)
    else:
        return np.exp(log_values)

