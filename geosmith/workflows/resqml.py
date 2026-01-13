"""RESQML (Reservoir Model XML) format support.

Migrated from geosuite.io.resqml_parser.
Layer 4: Workflows - I/O operations.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import RESQML libraries
try:
    import resqpy.model as rq
    import resqpy.grid as rqg
    import resqpy.property as rqp

    RESQPY_AVAILABLE = True
except ImportError:
    RESQPY_AVAILABLE = False
    rq = None  # type: ignore
    rqg = None  # type: ignore
    rqp = None  # type: ignore
    logger.warning(
        "resqpy not available. RESQML support requires resqpy. "
        "Install with: pip install resqpy"
    )


class ResqmlParser:
    """Parser for RESQML reservoir modeling files.

    Supports reading grid geometries, properties, and well trajectories
    from RESQML v2.0+ files.

    Example:
        >>> from geosmith.workflows.resqml import ResqmlParser
        >>>
        >>> parser = ResqmlParser()
        >>> grid_data = parser.load_grid('model.epc')
        >>> properties = parser.load_properties('model.epc', 'porosity')
    """

    def __init__(self):
        """Initialize RESQML parser."""
        if not RESQPY_AVAILABLE:
            raise ImportError(
                "resqpy is required for RESQML support. "
                "Install with: pip install resqpy"
            )
        self.model = None

    def load_model(self, epc_path: Union[str, Path]) -> None:
        """Load RESQML model from EPC file.

        Args:
            epc_path: Path to RESQML .epc file.

        Example:
            >>> from geosmith.workflows.resqml import ResqmlParser
            >>>
            >>> parser = ResqmlParser()
            >>> parser.load_model('model.epc')
        """
        epc_path = Path(epc_path)
        if not epc_path.exists():
            raise FileNotFoundError(f"RESQML file not found: {epc_path}")

        self.model = rq.Model(epc_file=str(epc_path))
        logger.info(f"Loaded RESQML model from {epc_path}")

    def load_grid(
        self,
        epc_path: Union[str, Path],
        grid_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load grid geometry from RESQML model.

        Args:
            epc_path: Path to RESQML .epc file.
            grid_name: Name of grid to load (uses first grid if not specified).

        Returns:
            Dictionary with grid data:
                - 'grid': Grid object
                - 'extent': Grid extent (nx, ny, nz)
                - 'origin': Grid origin coordinates
                - 'cell_centers': Cell center coordinates (sampled for large grids)
                - 'title': Grid title

        Example:
            >>> from geosmith.workflows.resqml import ResqmlParser
            >>>
            >>> parser = ResqmlParser()
            >>> grid_data = parser.load_grid('model.epc')
            >>> print(f"Grid extent: {grid_data['extent']}")
        """
        if self.model is None:
            self.load_model(epc_path)

        # Get grid(s)
        grids = self.model.grids()
        if not grids:
            raise ValueError("No grids found in RESQML model")

        if grid_name:
            grid = next((g for g in grids if g.title == grid_name), None)
            if grid is None:
                raise ValueError(f"Grid '{grid_name}' not found")
        else:
            grid = grids[0]

        # Extract grid information
        extent = (grid.nk, grid.nj, grid.ni)  # (nz, ny, nx)

        # Get origin
        origin = grid.origin if hasattr(grid, "origin") else None

        # Get cell centers (sample for large grids)
        try:
            centers = grid.centre_point()
            # For large grids, sample centers
            if centers.size > 1e6:
                sample_indices = np.random.choice(
                    centers.shape[0],
                    size=min(10000, centers.shape[0]),
                    replace=False,
                )
                centers = centers[sample_indices]
        except Exception as e:
            logger.warning(f"Could not extract cell centers: {e}")
            centers = None

        return {
            "grid": grid,
            "extent": extent,
            "origin": origin,
            "cell_centers": centers,
            "title": grid.title,
        }

    def load_properties(
        self,
        epc_path: Union[str, Path],
        property_name: Optional[str] = None,
        property_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load property arrays from RESQML model.

        Args:
            epc_path: Path to RESQML .epc file.
            property_name: Name of property to load.
            property_kind: Kind of property (e.g., 'porosity',
                'permeability', 'saturation').

        Returns:
            Dictionary with property data:
                - 'values': Property values array
                - 'name': Property name
                - 'kind': Property kind
                - 'uom': Unit of measure

        Example:
            >>> from geosmith.workflows.resqml import ResqmlParser
            >>>
            >>> parser = ResqmlParser()
            >>> porosity = parser.load_properties('model.epc', property_kind='porosity')
            >>> print(
            ...     f"Porosity range: {porosity['values'].min():.3f} - "
            ...     f"{porosity['values'].max():.3f}"
            ... )
        """
        if self.model is None:
            self.load_model(epc_path)

        # Get properties
        properties = self.model.properties()
        if not properties:
            raise ValueError("No properties found in RESQML model")

        # Filter by name or kind
        filtered_props = properties
        if property_name:
            filtered_props = [p for p in filtered_props if p.title == property_name]
        if property_kind:
            filtered_props = [
                p
                for p in filtered_props
                if hasattr(p, "property_kind") and p.property_kind == property_kind
            ]

        if not filtered_props:
            raise ValueError(
                f"No properties found matching name={property_name}, "
                f"kind={property_kind}"
            )

        prop = filtered_props[0]

        # Get property values
        try:
            values = prop.array_ref()
        except Exception as e:
            logger.warning(f"Could not extract property values: {e}")
            values = None

        return {
            "values": values,
            "name": prop.title,
            "kind": getattr(prop, "property_kind", None),
            "uom": getattr(prop, "uom", None),
        }


def read_resqml_grid(epc_path: Union[str, Path]) -> Dict[str, Any]:
    """Load grid geometry from RESQML model.

    Convenience function for loading RESQML grids.

    Args:
        epc_path: Path to RESQML .epc file.

    Returns:
        Dictionary with grid data.

    Example:
        >>> from geosmith.workflows.resqml import read_resqml_grid
        >>>
        >>> grid_data = read_resqml_grid('model.epc')
        >>> print(f"Grid extent: {grid_data['extent']}")
    """
    parser = ResqmlParser()
    return parser.load_grid(epc_path)


def read_resqml_properties(
    epc_path: Union[str, Path],
    property_name: Optional[str] = None,
    property_kind: Optional[str] = None,
) -> Dict[str, Any]:
    """Load property arrays from RESQML model.

    Convenience function for loading RESQML properties.

    Args:
        epc_path: Path to RESQML .epc file.
        property_name: Name of property to load.
        property_kind: Kind of property.

    Returns:
        Dictionary with property data.

    Example:
        >>> from geosmith.workflows.resqml import read_resqml_properties
        >>>
        >>> porosity = read_resqml_properties('model.epc', property_kind='porosity')
        >>> print(f"Porosity shape: {porosity['values'].shape}")
    """
    parser = ResqmlParser()
    return parser.load_properties(epc_path, property_name, property_kind)
