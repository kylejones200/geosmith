"""Block model generation task.

Migrated from geosuite.mining.block_model.
Layer 3: Tasks - User intent translation.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from geosmith.objects.pointset import PointSet

logger = logging.getLogger(__name__)


def create_block_model_grid(
    coords: np.ndarray,
    block_size_xy: float = 25.0,
    block_size_z: float = 10.0,
    bounds: Optional[Dict[str, float]] = None,
    quantile_padding: float = 0.05,
) -> Tuple[np.ndarray, Dict[str, Union[int, float, Tuple[float, float]]]]:
    """Create a 3D block model grid from sample coordinates.

    Generates a regular 3D grid with specified block sizes, suitable for
    mine planning applications. Grid bounds can be specified or auto-computed
    from data with optional padding.

    Args:
        coords: Sample coordinates (n_samples, 3) - [X, Y, Z].
        block_size_xy: Block size in X and Y directions (meters), default 25.0.
        block_size_z: Block size in Z direction (meters), default 10.0.
        bounds: Optional dictionary with keys 'x_min', 'x_max', 'y_min', 'y_max',
                'z_min', 'z_max'. If None, computed from data.
        quantile_padding: Padding as quantile (0-1) if bounds not specified.
                         Uses 5th and 95th percentiles by default, default 0.05.

    Returns:
        Tuple of (grid_coords, grid_info):
            - grid_coords: Grid coordinates (n_blocks, 3) - block centroids
            - grid_info: Dictionary with grid metadata:
                - 'nx', 'ny', 'nz': Grid dimensions
                - 'n_blocks': Total number of blocks
                - 'x_range', 'y_range', 'z_range': Coordinate ranges
                - 'block_size_xy', 'block_size_z': Block sizes

    Example:
        >>> from geosmith.tasks.blockmodeltask import create_block_model_grid
        >>> import numpy as np
        >>>
        >>> coords = np.random.rand(100, 3) * 1000
        >>> grid, info = create_block_model_grid(
        ...     coords, block_size_xy=25, block_size_z=10
        ... )
        >>> print(
        ...     f"Grid: {info['nx']} × {info['ny']} × {info['nz']} = "
        ...     f"{info['n_blocks']:,} blocks"
        ... )
    """
    coords = np.asarray(coords, dtype=np.float64)

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("Coordinates must be 2D array with 3 columns (X, Y, Z)")

    if len(coords) == 0:
        raise ValueError("Coordinates cannot be empty")

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
    G_x, G_y, G_z = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")

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

    logger.info(f"Created block model grid: {nx} × {ny} × {nz} = {n_blocks:,} blocks")

    return grid_coords, grid_info


class BlockModelTask:
    """Task for block model generation and processing.

    Translates user intent for block model operations into primitive calls.
    """

    def __init__(
        self,
        block_size_xy: float = 25.0,
        block_size_z: float = 10.0,
        quantile_padding: float = 0.05,
    ):
        """Initialize BlockModelTask.

        Args:
            block_size_xy: Block size in X and Y directions (meters), default 25.0.
            block_size_z: Block size in Z direction (meters), default 10.0.
            quantile_padding: Padding as quantile (0-1) for bounds calculation,
                default 0.05.
        """
        self.block_size_xy = block_size_xy
        self.block_size_z = block_size_z
        self.quantile_padding = quantile_padding

    def create_grid(
        self,
        coords: np.ndarray,
        bounds: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Union[int, float, Tuple[float, float]]]]:
        """Create block model grid from sample coordinates.

        Args:
            coords: Sample coordinates (n_samples, 3) - [X, Y, Z].
            bounds: Optional dictionary with bounds. If None, computed from data.

        Returns:
            Tuple of (grid_coords, grid_info).
        """
        return create_block_model_grid(
            coords,
            block_size_xy=self.block_size_xy,
            block_size_z=self.block_size_z,
            bounds=bounds,
            quantile_padding=self.quantile_padding,
        )
