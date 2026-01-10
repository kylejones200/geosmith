"""Clustering task for facies grouping.

Migrated from geosuite.ml.clustering.
Layer 3: Tasks - User intent translation.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from geosmith.objects.geotable import GeoTable
from geosmith.primitives.base import BaseEstimator

logger = logging.getLogger(__name__)

# Optional scikit-learn dependency
try:
    from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    KMeans = None  # type: ignore
    DBSCAN = None  # type: ignore
    AgglomerativeClustering = None  # type: ignore
    StandardScaler = None  # type: ignore
    Pipeline = None  # type: ignore


class FaciesClusterer(BaseEstimator):
    """Clustering pipeline for facies identification from well log data.

    Supports multiple clustering algorithms and provides a consistent API
    for facies grouping workflows.

    Example:
        >>> from geosmith.tasks.clusteringtask import FaciesClusterer
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> # Create sample well log data
        >>> df = pd.DataFrame({
        ...     'GR': np.random.normal(60, 15, 1000),
        ...     'RHOB': np.random.normal(2.5, 0.2, 1000),
        ...     'NPHI': np.random.normal(0.2, 0.1, 1000)
        ... })
        >>>
        >>> clusterer = FaciesClusterer(method='kmeans', n_clusters=5)
        >>> labels = clusterer.fit_predict(df)
        >>> print(f"Found {len(set(labels))} clusters")
    """

    def __init__(
        self,
        method: str = "kmeans",
        n_clusters: Optional[int] = None,
        scale_features: bool = True,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        """Initialize the facies clusterer.

        Args:
            method: Clustering method ('kmeans', 'dbscan', 'hierarchical').
            n_clusters: Number of clusters (required for kmeans/hierarchical, ignored for dbscan).
            scale_features: Whether to scale features before clustering, default True.
            random_state: Random seed for reproducibility, default 42.
            **kwargs: Additional parameters passed to clustering algorithm.
        """
        super().__init__()
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for clustering. "
                "Install with: pip install geosmith[ml] or pip install scikit-learn"
            )

        self.method = method.lower()
        self.n_clusters = n_clusters
        self.scale_features = scale_features
        self.random_state = random_state
        self.kwargs = kwargs

        self.scaler = StandardScaler() if scale_features else None
        self.model = None
        self.feature_names: Optional[List[str]] = None
        self.cluster_labels_: Optional[np.ndarray] = None

        self._build_model()

    def _build_model(self) -> None:
        """Build the clustering model based on method."""
        if not SKLEARN_AVAILABLE:
            return

        model_configs = {
            "kmeans": lambda: KMeans(
                n_clusters=self.n_clusters or 5,
                random_state=self.random_state,
                n_init=10,
                **self.kwargs,
            ),
            "dbscan": lambda: DBSCAN(
                eps=self.kwargs.get("eps", 0.5),
                min_samples=self.kwargs.get("min_samples", 5),
                **{
                    k: v
                    for k, v in self.kwargs.items()
                    if k not in ["eps", "min_samples"]
                },
            ),
            "hierarchical": lambda: AgglomerativeClustering(
                n_clusters=self.n_clusters or 5,
                linkage=self.kwargs.get("linkage", "ward"),
                **{k: v for k, v in self.kwargs.items() if k != "linkage"},
            ),
        }

        if self.method not in model_configs:
            raise ValueError(
                f"Unknown method: {self.method}. "
                f"Choose: {', '.join(model_configs.keys())}"
            )

        if self.method in ["kmeans", "hierarchical"] and self.n_clusters is None:
            raise ValueError(f"n_clusters must be specified for {self.method}")

        self.model = model_configs[self.method]()

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> "FaciesClusterer":
        """Fit the clustering model to data.

        Args:
            X: Feature array or DataFrame.
            y: Ignored (clustering is unsupervised), optional.

        Returns:
            self.

        Example:
            >>> from geosmith.tasks.clusteringtask import FaciesClusterer
            >>> import pandas as pd
            >>>
            >>> df = pd.DataFrame({'GR': [60, 80, 40], 'RHOB': [2.5, 2.3, 2.7]})
            >>> clusterer = FaciesClusterer(method='kmeans', n_clusters=3)
            >>> clusterer.fit(df)
            FaciesClusterer(...)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for clustering. "
                "Install with: pip install scikit-learn"
            )

        # For clustering, y is not used but we accept it for BaseEstimator compatibility
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_array = X.values
        else:
            X_array = np.asarray(X)
            self.feature_names = [
                f"feature_{i}" for i in range(X_array.shape[1])
            ]

        if len(X_array) == 0:
            raise ValueError("Input data must not be empty")

        if self.scale_features and self.scaler is not None:
            X_scaled = self.scaler.fit_transform(X_array)
        else:
            X_scaled = X_array

        if self.model is None:
            raise ValueError("Model not initialized")

        self.model.fit(X_scaled)
        self.cluster_labels_ = self.model.labels_

        n_clusters = len(set(self.cluster_labels_)) - (
            1 if -1 in self.cluster_labels_ else 0
        )
        logger.info(f"Fitted {self.method} clustering with {n_clusters} clusters")
        self._fitted = True

        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict cluster labels for new data.

        Args:
            X: Feature array or DataFrame.

        Returns:
            Cluster labels array.

        Example:
            >>> from geosmith.tasks.clusteringtask import FaciesClusterer
            >>> import pandas as pd
            >>>
            >>> df_train = pd.DataFrame({'GR': [60, 80, 40], 'RHOB': [2.5, 2.3, 2.7]})
            >>> clusterer = FaciesClusterer(method='kmeans', n_clusters=3)
            >>> clusterer.fit(df_train)
            >>>
            >>> df_test = pd.DataFrame({'GR': [65, 75], 'RHOB': [2.45, 2.35]})
            >>> labels = clusterer.predict(df_test)
            >>> print(f"Predicted labels: {labels}")
        """
        if not self._fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")

        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.asarray(X)

        if self.scale_features:
            if self.scaler is None:
                raise ValueError("Scaler must be fitted before prediction")
            X_scaled = self.scaler.transform(X_array)
        else:
            X_scaled = X_array

        if self.method == "dbscan":
            return self.model.fit_predict(X_scaled)
        else:
            return self.model.predict(X_scaled)

    def fit_predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Fit the model and predict cluster labels.

        Args:
            X: Feature array or DataFrame.

        Returns:
            Cluster labels array.
        """
        return self.fit(X).cluster_labels_ or np.array([])


def cluster_facies(
    df: Union[pd.DataFrame, GeoTable],
    feature_cols: List[str],
    method: str = "kmeans",
    n_clusters: Optional[int] = None,
    scale_features: bool = True,
    random_state: int = 42,
    **kwargs: Any,
) -> pd.Series:
    """Cluster facies from well log data.

    Convenience function for quick facies clustering.

    Args:
        df: DataFrame or GeoTable with well log data.
        feature_cols: List of column names to use as features.
        method: Clustering method ('kmeans', 'dbscan', 'hierarchical'), default 'kmeans'.
        n_clusters: Number of clusters (required for kmeans/hierarchical), optional.
        scale_features: Whether to scale features, default True.
        random_state: Random seed, default 42.
        **kwargs: Additional clustering parameters.

    Returns:
        Series with cluster labels.

    Example:
        >>> from geosmith.tasks.clusteringtask import cluster_facies
        >>> import pandas as pd
        >>>
        >>> df = pd.DataFrame({
        ...     'GR': [60, 80, 40, 70, 50],
        ...     'RHOB': [2.5, 2.3, 2.7, 2.4, 2.6],
        ...     'NPHI': [0.2, 0.15, 0.25, 0.18, 0.22]
        ... })
        >>> labels = cluster_facies(df, ['GR', 'RHOB', 'NPHI'], method='kmeans', n_clusters=3)
        >>> print(f"Cluster labels: {labels.values}")
    """
    # Extract DataFrame if GeoTable
    if isinstance(df, GeoTable):
        df_data = df.data.copy()
    else:
        df_data = df.copy()

    clusterer = FaciesClusterer(
        method=method,
        n_clusters=n_clusters,
        scale_features=scale_features,
        random_state=random_state,
        **kwargs,
    )

    labels = clusterer.fit_predict(df_data[feature_cols])

    return pd.Series(labels, index=df_data.index, name="facies_cluster")


def find_optimal_clusters(
    X: Union[np.ndarray, pd.DataFrame],
    method: str = "kmeans",
    max_clusters: int = 10,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Find optimal number of clusters using elbow method (for kmeans).

    Args:
        X: Feature array or DataFrame.
        method: Clustering method (currently only 'kmeans' supported), default 'kmeans'.
        max_clusters: Maximum number of clusters to test, default 10.
        random_state: Random seed, default 42.

    Returns:
        Dictionary with cluster counts and inertia/silhouette scores:
            - 'n_clusters': List of cluster counts tested
            - 'inertias': List of inertia values
            - 'optimal_n': Optimal number of clusters (elbow point)

    Example:
        >>> from geosmith.tasks.clusteringtask import find_optimal_clusters
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> df = pd.DataFrame(np.random.randn(100, 4))
        >>> result = find_optimal_clusters(df, max_clusters=10)
        >>> print(f"Optimal clusters: {result['optimal_n']}")
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required for optimal cluster finding. "
            "Install with: pip install scikit-learn"
        )

    if method != "kmeans":
        raise ValueError(
            "Optimal cluster finding currently only supports kmeans"
        )

    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = np.asarray(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_array)

    inertias = []
    n_clusters_range = range(2, max_clusters + 1)

    for n in n_clusters_range:
        kmeans = KMeans(n_clusters=n, random_state=random_state, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)

    return {
        "n_clusters": list(n_clusters_range),
        "inertias": inertias,
        "optimal_n": _find_elbow(inertias, list(n_clusters_range)),
    }


def _find_elbow(inertias: List[float], n_clusters: List[int]) -> int:
    """Find elbow point in inertia curve.

    Uses the point of maximum curvature.

    Args:
        inertias: List of inertia values.
        n_clusters: List of cluster counts.

    Returns:
        Optimal number of clusters (elbow point).
    """
    if len(inertias) < 3:
        return n_clusters[len(inertias) // 2]

    inertias_arr = np.array(inertias)
    n_clusters_arr = np.array(n_clusters)

    first_point = np.array([n_clusters_arr[0], inertias_arr[0]])
    last_point = np.array([n_clusters_arr[-1], inertias_arr[-1]])

    distances = []
    for i in range(len(n_clusters_arr)):
        point = np.array([n_clusters_arr[i], inertias_arr[i]])
        d = np.abs(
            np.cross(last_point - first_point, first_point - point)
        ) / np.linalg.norm(last_point - first_point)
        distances.append(d)

    elbow_idx = np.argmax(distances)
    return int(n_clusters_arr[elbow_idx])

