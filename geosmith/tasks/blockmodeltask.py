"""BlockModelTask: Block model generation for 3D spatial data.

Migrated from geosuite.mining.block_model.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from geosmith.objects.pointset import PointSet
from geosmith.objects.rastergrid import RasterGrid
from geosmith.objects.geoindex import GeoIndex
from geosmith.primitives.interpolation import idw_interpolate

logger = logging.getLogger(__name__)


class BlockModelTask:
    """Task for creating 3D block models from point data.

    Supports grid generation, grade estimation via IDW, and export.
    """

    def __init__(self, crs: Optional[str] = None):
        """Initialize BlockModelTask.

        Args:
            crs: Optional default CRS for output models.
        """
        self.crs = crs

    def create_block_model_grid(
        self,
        sample_points: PointSet,
        block_size_xy: float = 25.0,
        block_size_z: float = 10.0,
        bounds: Optional[Dict[str, float]] = None,
        quantile_padding: float = 0.05,
    ) -> Tuple[PointSet, Dict[str, any]]:
        """Create a 3D block model grid from sample coordinates.

        Generates a regular 3D grid with specified block sizes, suitable for
        mine planning applications.

        Args:
            sample_points: PointSet with sample locations (n_samples, 3).
            block_size_xy: Block size in X and Y directions (meters).
            block_size_z: Block size in Z direction (meters).
            bounds: Optional dictionary with keys 'x_min', 'x_max', 'y_min', 'y_max',
                    'z_min', 'z_max'. If None, computed from data.
            quantile_padding: Padding as quantile (0-1) if bounds not specified.

        Returns:
            Tuple of (grid_points, grid_info):
                - grid_points: PointSet with grid block centroids (n_blocks, 3).
                - grid_info: Dictionary with grid metadata.

        Example:
            >>> from geosmith import PointSet
            >>> import numpy as np
            >>> 
            >>> samples = PointSet(coordinates=np.array([[100, 200, 50], [150, 250, 60]]))
            >>> task = BlockModelTask()
            >>> grid, info = task.create_block_model_grid(samples, block_size_xy=25, block_size_z=10)
            >>> print(f"Grid: {info['nx']} × {info['ny']} × {info['nz']} = {info['n_blocks']:,} blocks")
        """
        coords = sample_points.coordinates

        if coords.shape[1] != 3:
            raise ValueError("Block models require 3D coordinates (X, Y, Z)")

        # Determine bounds
        if bounds is None:
            # Use quantiles to exclude outliers
            x_min, x_max = np.quantile(
                coords[:, 0], [quantile_padding, 1 - quantile_padding]
            )
            y_min, y_max = np.quantile(
                coords[:, 1], [quantile_padding, 1 - quantile_padding]
            )
            z_min, z_max = np.quantile(
                coords[:, 2], [quantile_padding, 1 - quantile_padding]
            )
        else:
            x_min = bounds["x_min"]
            x_max = bounds["x_max"]
            y_min = bounds["y_min"]
            y_max = bounds["y_max"]
            z_min = bounds["z_min"]
            z_max = bounds["z_max"]

        # Calculate grid dimensions
        nx = int(np.ceil((x_max - x_min) / block_size_xy))
        ny = int(np.ceil((y_max - y_min) / block_size_xy))
        nz = int(np.ceil((z_max - z_min) / block_size_z))

        # Adjust bounds to be evenly divisible by block size
        x_max = x_min + nx * block_size_xy
        y_max = y_min + ny * block_size_xy
        z_max = z_min + nz * block_size_z

        # Create grid coordinates (block centroids)
        x_coords = np.linspace(x_min, x_max, nx) + block_size_xy / 2
        y_coords = np.linspace(y_min, y_max, ny) + block_size_xy / 2
        z_coords = np.linspace(z_min, z_max, nz) + block_size_z / 2

        # Create 3D meshgrid
        G_x, G_y, G_z = np.meshgrid(
            x_coords, y_coords, z_coords, indexing="ij"
        )

        # Flatten to coordinate matrix
        grid_coords = np.column_stack([G_x.ravel(), G_y.ravel(), G_z.ravel()])

        n_blocks = len(grid_coords)

        grid_info = {
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "n_blocks": n_blocks,
            "x_range": (x_min, x_max),
            "y_range": (y_min, y_max),
            "z_range": (z_min, z_max),
            "block_size_xy": block_size_xy,
            "block_size_z": block_size_z,
        }

        logger.info(
            f"Created block model grid: {nx} × {ny} × {nz} = {n_blocks:,} blocks"
        )

        # Create PointSet for grid
        index = GeoIndex(
            crs=self.crs or (sample_points.index.crs if sample_points.index else None),
            bounds=(x_min, y_min, x_max, y_max),
        )
        grid_points = PointSet(coordinates=grid_coords, index=index)

        return grid_points, grid_info

    def estimate_grades(
        self,
        sample_points: PointSet,
        sample_values: np.ndarray,
        grid_points: PointSet,
        k: int = 16,
        power: float = 2.0,
    ) -> np.ndarray:
        """Estimate grades at grid points using IDW interpolation.

        Args:
            sample_points: PointSet with sample locations.
            sample_values: Sample values (n_samples,).
            grid_points: PointSet with grid block centroids.
            k: Number of nearest neighbors for IDW.
            power: IDW exponent.

        Returns:
            Estimated values at grid points (n_blocks,).
        """
        logger.info(
            f"Estimating grades for {len(grid_points.coordinates)} blocks using IDW"
        )

        return idw_interpolate(
            sample_points, sample_values, grid_points, k=k, power=power
        )

    def create_block_model(
        self,
        sample_points: PointSet,
        sample_values: np.ndarray,
        block_size_xy: float = 25.0,
        block_size_z: float = 10.0,
        k: int = 16,
        power: float = 2.0,
        **kwargs,
    ) -> pd.DataFrame:
        """Create complete block model with grade estimates.

        Args:
            sample_points: PointSet with sample locations.
            sample_values: Sample values (n_samples,).
            block_size_xy: Block size in X and Y directions.
            block_size_z: Block size in Z direction.
            k: Number of nearest neighbors for IDW.
            power: IDW exponent.
            **kwargs: Additional arguments passed to create_block_model_grid.

        Returns:
            DataFrame with block model data (x, y, z, grade).
        """
        logger.info("Creating block model")

        # Create grid
        grid_points, grid_info = self.create_block_model_grid(
            sample_points,
            block_size_xy=block_size_xy,
            block_size_z=block_size_z,
            **kwargs,
        )

        # Estimate grades
        grades = self.estimate_grades(
            sample_points, sample_values, grid_points, k=k, power=power
        )

        # Create DataFrame
        block_model = pd.DataFrame(
            {
                "x": grid_points.coordinates[:, 0],
                "y": grid_points.coordinates[:, 1],
                "z": grid_points.coordinates[:, 2],
                "grade": grades,
            }
        )

        logger.info(f"Created block model with {len(block_model)} blocks")
        return block_model

