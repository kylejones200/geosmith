"""Block model generation task.

Migrated from geosuite.mining.block_model.
Layer 3: Tasks - User intent translation.

Enhanced with:
- Sub-blocking (dividing blocks into smaller sub-blocks)
- Rotated grids (support for rotated coordinate systems)
- Variable block sizes (different sizes in different regions)
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from geosmith.objects.pointset import PointSet
from geosmith.utils.errors import ParameterError

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


def create_rotated_block_model_grid(
    coords: np.ndarray,
    block_size_xy: float = 25.0,
    block_size_z: float = 10.0,
    rotation_angle: float = 0.0,
    rotation_center: Optional[np.ndarray] = None,
    bounds: Optional[Dict[str, float]] = None,
    quantile_padding: float = 0.05,
) -> Tuple[np.ndarray, Dict[str, Union[int, float, Tuple[float, float]]]]:
    """Create a rotated 3D block model grid.

    Creates a block model grid that can be rotated around a center point.
    Useful for aligning grids with geological structures or mine layouts.

    Args:
        coords: Sample coordinates (n_samples, 3) - [X, Y, Z].
        block_size_xy: Block size in X and Y directions (meters).
        block_size_z: Block size in Z direction (meters).
        rotation_angle: Rotation angle in degrees (counterclockwise).
        rotation_center: Center point for rotation (3,). If None, uses data centroid.
        bounds: Optional dictionary with bounds in rotated coordinate system.
        quantile_padding: Padding as quantile (0-1) if bounds not specified.

    Returns:
        Tuple of (grid_coords, grid_info) with rotated grid coordinates.
    """
    coords = np.asarray(coords, dtype=np.float64)

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("Coordinates must be 2D array with 3 columns (X, Y, Z)")

    if len(coords) == 0:
        raise ValueError("Coordinates cannot be empty")

    # Determine rotation center
    if rotation_center is None:
        rotation_center = np.mean(coords, axis=0)
    else:
        rotation_center = np.asarray(rotation_center, dtype=np.float64)
        if rotation_center.shape != (3,):
            raise ValueError("rotation_center must be array of shape (3,)")

    # Convert angle to radians
    angle_rad = np.deg2rad(rotation_angle)

    # Rotation matrix (around Z-axis, keeping Z unchanged)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rotation_matrix = np.array(
        [
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1],
        ]
    )

    # Rotate coordinates to align with grid axes
    coords_centered = coords - rotation_center
    coords_rotated = (rotation_matrix @ coords_centered.T).T + rotation_center

    # Create grid in rotated coordinate system
    grid_coords_rotated, grid_info = create_block_model_grid(
        coords_rotated,
        block_size_xy=block_size_xy,
        block_size_z=block_size_z,
        bounds=bounds,
        quantile_padding=quantile_padding,
    )

    # Rotate grid coordinates back to original coordinate system
    grid_coords_centered = grid_coords_rotated - rotation_center
    rotation_matrix_inv = rotation_matrix.T  # Inverse of rotation matrix
    grid_coords = (rotation_matrix_inv @ grid_coords_centered.T).T + rotation_center

    # Add rotation info to grid_info
    grid_info["rotation_angle"] = rotation_angle
    grid_info["rotation_center"] = rotation_center

    logger.info(
        f"Created rotated block model grid: {grid_info['nx']} × "
        f"{grid_info['ny']} × {grid_info['nz']} = {grid_info['n_blocks']:,} blocks "
        f"(rotated {rotation_angle:.1f}°)"
    )

    return grid_coords, grid_info


def create_sub_blocked_grid(
    parent_grid_coords: np.ndarray,
    parent_grid_info: Dict[str, Union[int, float, Tuple[float, float]]],
    sub_divisions: Tuple[int, int, int] = (2, 2, 2),
) -> Tuple[np.ndarray, Dict[str, Union[int, float, Tuple[float, float]]]]:
    """Create sub-blocked grid by dividing parent blocks into smaller sub-blocks.

    Useful for refining block models in specific regions or for detailed
    analysis near boundaries.

    Args:
        parent_grid_coords: Parent grid coordinates (n_blocks, 3).
        parent_grid_info: Parent grid info dictionary.
        sub_divisions: Number of subdivisions in (x, y, z) directions.

    Returns:
        Tuple of (sub_grid_coords, sub_grid_info).
    """
    parent_grid_coords = np.asarray(parent_grid_coords, dtype=np.float64)

    if parent_grid_coords.ndim != 2 or parent_grid_coords.shape[1] != 3:
        raise ValueError("Parent grid coordinates must be 2D array with 3 columns")

    if len(parent_grid_coords) == 0:
        raise ValueError("Parent grid coordinates cannot be empty")

    sub_nx, sub_ny, sub_nz = sub_divisions

    if sub_nx < 1 or sub_ny < 1 or sub_nz < 1:
        raise ParameterError(
            "sub_divisions",
            sub_divisions,
            constraint="All values must be >= 1",
            suggestion="Use (2, 2, 2) or higher for sub-blocking",
        )

    # Get parent block sizes
    parent_block_size_xy = parent_grid_info["block_size_xy"]
    parent_block_size_z = parent_grid_info["block_size_z"]

    # Calculate sub-block sizes
    sub_block_size_xy = parent_block_size_xy / sub_nx
    sub_block_size_z = parent_block_size_z / sub_nz

    # Create sub-blocks for each parent block
    sub_grid_coords_list = []

    for parent_coord in parent_grid_coords:
        # Calculate parent block bounds
        x_center, y_center, z_center = parent_coord
        x_min = x_center - parent_block_size_xy / 2
        x_max = x_center + parent_block_size_xy / 2
        y_min = y_center - parent_block_size_xy / 2
        y_max = y_center + parent_block_size_xy / 2
        z_min = z_center - parent_block_size_z / 2
        z_max = z_center + parent_block_size_z / 2

        # Create sub-block coordinates
        x_sub = np.linspace(
            x_min + sub_block_size_xy / 2,
            x_max - sub_block_size_xy / 2,
            sub_nx,
        )
        y_sub = np.linspace(
            y_min + sub_block_size_xy / 2,
            y_max - sub_block_size_xy / 2,
            sub_ny,
        )
        z_sub = np.linspace(
            z_min + sub_block_size_z / 2,
            z_max - sub_block_size_z / 2,
            sub_nz,
        )

        # Create meshgrid for sub-blocks
        G_x, G_y, G_z = np.meshgrid(x_sub, y_sub, z_sub, indexing="ij")
        sub_blocks = np.column_stack([G_x.ravel(), G_y.ravel(), G_z.ravel()])
        sub_grid_coords_list.append(sub_blocks)

    # Combine all sub-blocks
    sub_grid_coords = np.vstack(sub_grid_coords_list)

    # Update grid info
    parent_nx = parent_grid_info["nx"]
    parent_ny = parent_grid_info["ny"]
    parent_nz = parent_grid_info["nz"]

    sub_grid_info = {
        "nx": parent_nx * sub_nx,
        "ny": parent_ny * sub_ny,
        "nz": parent_nz * sub_nz,
        "n_blocks": len(sub_grid_coords),
        "x_range": parent_grid_info["x_range"],
        "y_range": parent_grid_info["y_range"],
        "z_range": parent_grid_info["z_range"],
        "block_size_xy": sub_block_size_xy,
        "block_size_z": sub_block_size_z,
        "parent_n_blocks": parent_grid_info["n_blocks"],
        "sub_divisions": sub_divisions,
    }

    logger.info(
        f"Created sub-blocked grid: {sub_grid_info['nx']} × "
        f"{sub_grid_info['ny']} × {sub_grid_info['nz']} = "
        f"{sub_grid_info['n_blocks']:,} sub-blocks "
        f"(from {parent_grid_info['n_blocks']:,} parent blocks)"
    )

    return sub_grid_coords, sub_grid_info


def create_variable_block_size_grid(
    coords: np.ndarray,
    block_size_regions: List[Dict[str, Union[float, Tuple[float, float]]]],
    default_block_size_xy: float = 25.0,
    default_block_size_z: float = 10.0,
    quantile_padding: float = 0.05,
) -> Tuple[np.ndarray, Dict[str, Union[int, float, Tuple[float, float]]]]:
    """Create block model grid with variable block sizes in different regions.

    Allows different block sizes in different spatial regions, useful for
    adaptive refinement or region-specific requirements.

    Args:
        coords: Sample coordinates (n_samples, 3) - [X, Y, Z].
        block_size_regions: List of region dictionaries, each with:
            - 'bounds': Dict with 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'
            - 'block_size_xy': Block size in X and Y directions
            - 'block_size_z': Block size in Z direction
        default_block_size_xy: Default block size for regions not specified.
        default_block_size_z: Default block size for Z direction.
        quantile_padding: Padding as quantile (0-1) if bounds not specified.

    Returns:
        Tuple of (grid_coords, grid_info) with combined grid.
    """
    coords = np.asarray(coords, dtype=np.float64)

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("Coordinates must be 2D array with 3 columns (X, Y, Z)")

    if len(coords) == 0:
        raise ValueError("Coordinates cannot be empty")

    # Determine overall bounds
    x_min, x_max = np.quantile(
        coords[:, 0], [quantile_padding, 1 - quantile_padding]
    )
    y_min, y_max = np.quantile(
        coords[:, 1], [quantile_padding, 1 - quantile_padding]
    )
    z_min, z_max = np.quantile(
        coords[:, 2], [quantile_padding, 1 - quantile_padding]
    )

    # Create grids for each region
    all_grid_coords = []
    grid_info_list = []

    for region in block_size_regions:
        region_bounds = region.get("bounds")
        region_block_size_xy = region.get("block_size_xy", default_block_size_xy)
        region_block_size_z = region.get("block_size_z", default_block_size_z)

        if region_bounds is None:
            # Use overall bounds
            region_bounds = {
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "z_min": z_min,
                "z_max": z_max,
            }

        # Create grid for this region
        region_coords, region_info = create_block_model_grid(
            coords,
            block_size_xy=region_block_size_xy,
            block_size_z=region_block_size_z,
            bounds=region_bounds,
            quantile_padding=0.0,  # Already using specified bounds
        )

        # Filter to only include blocks within region bounds
        mask = (
            (region_coords[:, 0] >= region_bounds["x_min"])
            & (region_coords[:, 0] <= region_bounds["x_max"])
            & (region_coords[:, 1] >= region_bounds["y_min"])
            & (region_coords[:, 1] <= region_bounds["y_max"])
            & (region_coords[:, 2] >= region_bounds["z_min"])
            & (region_coords[:, 2] <= region_bounds["z_max"])
        )

        all_grid_coords.append(region_coords[mask])
        grid_info_list.append(region_info)

    # Combine all grids (remove duplicates if any)
    combined_grid = np.vstack(all_grid_coords)

    # Remove duplicate blocks (if regions overlap)
    # Use simple distance-based duplicate detection
    from geosmith.utils.optional_imports import optional_import_single

    SCIPY_AVAILABLE, _ = optional_import_single("scipy.spatial.distance", "cdist")
    if SCIPY_AVAILABLE:
        from scipy.spatial.distance import cdist  # type: ignore

        # Simple duplicate detection using distance matrix
        distances = cdist(combined_grid, combined_grid)
        np.fill_diagonal(distances, np.inf)
        is_duplicate = np.any(distances < 1e-6, axis=1)
    else:
        # Fallback: no duplicate removal if scipy not available
        is_duplicate = np.zeros(len(combined_grid), dtype=bool)

    # Keep only non-duplicates (keep first occurrence)
    unique_mask = ~is_duplicate
    final_grid_coords = combined_grid[unique_mask]

    # Create combined grid info
    combined_info = {
        "n_blocks": len(final_grid_coords),
        "x_range": (x_min, x_max),
        "y_range": (y_min, y_max),
        "z_range": (z_min, z_max),
        "n_regions": len(block_size_regions),
        "variable_block_sizes": True,
    }

    logger.info(
        f"Created variable block size grid: {combined_info['n_blocks']:,} blocks "
        f"across {len(block_size_regions)} regions"
    )

    return final_grid_coords, combined_info
