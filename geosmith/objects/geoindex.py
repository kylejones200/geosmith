"""GeoIndex: Coordinate reference system, bounds, and axis order."""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class GeoIndex:
    """Immutable representation of spatial reference information.

    Attributes:
        crs: Coordinate reference system identifier (e.g., 'EPSG:4326').
        bounds: Bounding box as (minx, miny, maxx, maxy).
        axis_order: Axis order, either 'xy' (lon, lat) or 'yx' (lat, lon).
            Defaults to 'xy'.
    """

    crs: Optional[str]
    bounds: Tuple[float, float, float, float]
    axis_order: str = "xy"

    def __post_init__(self) -> None:
        """Validate GeoIndex parameters."""
        if self.axis_order not in ("xy", "yx"):
            raise ValueError(f"axis_order must be 'xy' or 'yx', got {self.axis_order}")

        if len(self.bounds) != 4:
            raise ValueError(f"bounds must have 4 elements, got {len(self.bounds)}")

        minx, miny, maxx, maxy = self.bounds
        if minx >= maxx:
            raise ValueError(f"Invalid bounds: minx ({minx}) >= maxx ({maxx})")
        if miny >= maxy:
            raise ValueError(f"Invalid bounds: miny ({miny}) >= maxy ({maxy})")

    def __repr__(self) -> str:
        """String representation."""
        crs_str = self.crs if self.crs else "None"
        return f"GeoIndex(crs={crs_str}, bounds={self.bounds}, axis_order={self.axis_order})"

