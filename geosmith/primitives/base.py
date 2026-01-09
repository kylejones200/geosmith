"""Base classes for primitives with tag system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class BaseObject(ABC):
    """Base class for all primitive objects.

    Attributes:
        params: Dictionary of parameters.
        tags: Dictionary of capability tags.
    """

    params: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize default tags."""
        if not self.tags:
            self.tags = {
                "supports_crs_transform": False,
                "requires_projected_crs": False,
                "supports_3d": False,
                "handles_nodata": False,
                "supports_raster": False,
                "supports_vector": False,
            }

    def clone(self, **kwargs: Any) -> "BaseObject":
        """Create a copy with updated parameters."""
        import copy

        new_params = {**self.params, **kwargs}
        new_obj = copy.deepcopy(self)
        new_obj.params = new_params
        return new_obj

    def __repr__(self) -> str:
        """String representation."""
        class_name = self.__class__.__name__
        return f"{class_name}(params={self.params}, tags={self.tags})"


@dataclass
class BaseEstimator(BaseObject):
    """Base class for estimators that have fit state."""

    _fitted: bool = field(default=False, init=False, repr=False)

    def fit(self, *args: Any, **kwargs: Any) -> "BaseEstimator":
        """Fit the estimator."""
        self._fitted = True
        return self

    @property
    def is_fitted(self) -> bool:
        """Check if estimator is fitted."""
        return self._fitted


@dataclass
class BaseTransformer(BaseObject):
    """Base class for transformers."""

    def transform(self, *args: Any, **kwargs: Any) -> Any:
        """Transform input data."""
        raise NotImplementedError("Subclasses must implement transform")


@dataclass
class BaseSpatialModel(BaseEstimator):
    """Base class for spatial models that can predict."""

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """Make predictions."""
        raise NotImplementedError("Subclasses must implement predict")


@dataclass
class BaseRasterModel(BaseEstimator):
    """Base class for raster models that can predict."""

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """Make predictions."""
        raise NotImplementedError("Subclasses must implement predict")

