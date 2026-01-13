"""DLIS (Digital Log Interchange Standard) format support.

Migrated from geosuite.io.dlis_parser.
Layer 4: Workflows - I/O operations.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import DLIS libraries
try:
    import dlisio

    DLISIO_AVAILABLE = True
except ImportError:
    DLISIO_AVAILABLE = False
    dlisio = None  # type: ignore
    logger.warning(
        "dlisio not available. DLIS support requires dlisio. "
        "Install with: pip install dlisio"
    )


class DlisParser:
    """Parser for DLIS well log files.

    Supports reading channels, frames, and well information
    from DLIS files.

    Example:
        >>> from geosmith.workflows.dlis import DlisParser
        >>>
        >>> parser = DlisParser()
        >>> df = parser.load_channels('log.dlis')
        >>> well_info = parser.get_well_info('log.dlis')
    """

    def __init__(self):
        """Initialize DLIS parser."""
        if not DLISIO_AVAILABLE:
            raise ImportError(
                "dlisio is required for DLIS support. "
                "Install with: pip install dlisio"
            )

    def load_channels(
        self,
        dlis_path: Union[str, Path],
        channel_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Load channel data from DLIS file.

        Args:
            dlis_path: Path to DLIS file.
            channel_names: Specific channels to load (loads all if not specified).

        Returns:
            DataFrame with channels as columns.

        Example:
            >>> from geosmith.workflows.dlis import DlisParser
            >>>
            >>> parser = DlisParser()
            >>> df = parser.load_channels(
            ...     'log.dlis', channel_names=['GR', 'RHOB', 'NPHI']
            ... )
            >>> print(f"Loaded {len(df)} samples with {len(df.columns)} channels")
        """
        dlis_path = Path(dlis_path)
        if not dlis_path.exists():
            raise FileNotFoundError(f"DLIS file not found: {dlis_path}")

        with dlisio.open(str(dlis_path)) as f:
            # Get all channels
            channels = f.channels()

            if not channels:
                raise ValueError("No channels found in DLIS file")

            # Filter by name if specified
            if channel_names:
                channels = {
                    name: ch for name, ch in channels.items() if name in channel_names
                }

            if not channels:
                raise ValueError(
                    f"None of the specified channels found: {channel_names}"
                )

            # Get frame to determine length
            frames = f.frames()
            if not frames:
                raise ValueError("No frames found in DLIS file")

            frame = list(frames.values())[0]
            n_samples = len(frame)

            # Build DataFrame
            data = {}

            for name, channel in channels.items():
                try:
                    # Read channel data
                    values = channel.curves()
                    if len(values) == n_samples:
                        data[name] = values
                    else:
                        logger.warning(
                            f"Channel {name} has length {len(values)}, "
                            f"expected {n_samples}. Skipping."
                        )
                except Exception as e:
                    logger.warning(f"Error reading channel {name}: {e}")

            if not data:
                raise ValueError("No valid channel data could be read")

            df = pd.DataFrame(data)

            # Add index as depth if available
            if "TDEP" in df.columns:
                df.index = df["TDEP"]
                df.index.name = "DEPTH"
            elif "DEPT" in df.columns:
                df.index = df["DEPT"]
                df.index.name = "DEPTH"

            return df

    def get_well_info(self, dlis_path: Union[str, Path]) -> Dict[str, Any]:
        """Extract well information from DLIS file.

        Args:
            dlis_path: Path to DLIS file.

        Returns:
            Dictionary with well information:
                - 'well_name': Well name
                - 'field': Field name
                - 'company': Company name
                - 'service_company': Service company
                - 'run_number': Run number

        Example:
            >>> from geosmith.workflows.dlis import DlisParser
            >>>
            >>> parser = DlisParser()
            >>> well_info = parser.get_well_info('log.dlis')
            >>> print(f"Well: {well_info.get('well_name', 'Unknown')}")
        """
        dlis_path = Path(dlis_path)
        if not dlis_path.exists():
            raise FileNotFoundError(f"DLIS file not found: {dlis_path}")

        with dlisio.open(str(dlis_path)) as f:
            # Get origin (well information)
            origins = f.origins()
            if not origins:
                return {}

            origin = list(origins.values())[0]

            info = {}

            # Extract well information
            if hasattr(origin, "well_name"):
                info["well_name"] = str(origin.well_name)
            if hasattr(origin, "field_name"):
                info["field"] = str(origin.field_name)
            if hasattr(origin, "producer_name"):
                info["company"] = str(origin.producer_name)
            if hasattr(origin, "service_company"):
                info["service_company"] = str(origin.service_company)
            if hasattr(origin, "run_number"):
                info["run_number"] = str(origin.run_number)

            return info

    def list_channels(self, dlis_path: Union[str, Path]) -> List[str]:
        """List all available channels in DLIS file.

        Args:
            dlis_path: Path to DLIS file.

        Returns:
            List of channel names.

        Example:
            >>> from geosmith.workflows.dlis import DlisParser
            >>>
            >>> parser = DlisParser()
            >>> channels = parser.list_channels('log.dlis')
            >>> print(f"Available channels: {', '.join(channels[:10])}")
        """
        dlis_path = Path(dlis_path)
        if not dlis_path.exists():
            raise FileNotFoundError(f"DLIS file not found: {dlis_path}")

        with dlisio.open(str(dlis_path)) as f:
            channels = f.channels()
            return list(channels.keys())


def read_dlis_file(
    dlis_path: Union[str, Path],
    channel_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load channel data from DLIS file.

    Convenience function for loading DLIS files.

    Args:
        dlis_path: Path to DLIS file.
        channel_names: Specific channels to load (loads all if not specified).

    Returns:
        DataFrame with channels as columns.

    Example:
        >>> from geosmith.workflows.dlis import read_dlis_file
        >>>
        >>> df = read_dlis_file('log.dlis', channel_names=['GR', 'RHOB', 'NPHI'])
        >>> print(f"Loaded {len(df)} samples")
    """
    parser = DlisParser()
    return parser.load_channels(dlis_path, channel_names=channel_names)
