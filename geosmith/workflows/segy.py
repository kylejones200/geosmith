"""SEG-Y file I/O for seismic data.

Migrated from geosuite.io.segy_loader.
Layer 4: Workflows - I/O operations.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import segyio
try:
    import segyio

    SEGYIO_AVAILABLE = True
except ImportError:
    SEGYIO_AVAILABLE = False
    segyio = None  # type: ignore


@dataclass
class SegySummary:
    """Summary information from SEG-Y file.

    Attributes:
        path: Path to SEG-Y file.
        n_traces: Number of traces.
        n_samples: Number of samples per trace.
        sample_rate_us: Sample rate in microseconds.
        text_header: Text header content.
        has_inline_crossline: Whether file has inline/crossline geometry.
        format: Data format code.
        sample_interval_us: Sample interval in microseconds.
    """

    path: str
    n_traces: int
    n_samples: int
    sample_rate_us: float
    text_header: str
    has_inline_crossline: bool
    format: int
    sample_interval_us: float


@dataclass
class TraceHeader:
    """Trace header information.

    Attributes:
        trace_number: Trace sequence number.
        inline: Optional inline number.
        crossline: Optional crossline number.
        x: Optional X coordinate.
        y: Optional Y coordinate.
        elevation: Optional elevation.
        source_depth: Optional source depth.
        receiver_elevation: Optional receiver elevation.
        source_x: Optional source X coordinate.
        source_y: Optional source Y coordinate.
        receiver_x: Optional receiver X coordinate.
        receiver_y: Optional receiver Y coordinate.
        offset: Optional offset.
        cdp_x: Optional CDP X coordinate.
        cdp_y: Optional CDP Y coordinate.
        cdp: Optional CDP number.
        raw_header: Raw header dictionary.
    """

    trace_number: int
    inline: Optional[int]
    crossline: Optional[int]
    x: Optional[float]
    y: Optional[float]
    elevation: Optional[float]
    source_depth: Optional[float]
    receiver_elevation: Optional[float]
    source_x: Optional[float]
    source_y: Optional[float]
    receiver_x: Optional[float]
    receiver_y: Optional[float]
    offset: Optional[float]
    cdp_x: Optional[float]
    cdp_y: Optional[float]
    cdp: Optional[int]
    raw_header: Dict[str, Any]


def read_segy_summary(path: str | Path) -> SegySummary:
    """Read summary information from SEG-Y file.

    Args:
        path: Path to SEG-Y file.

    Returns:
        SegySummary with file information.

    Raises:
        ImportError: If segyio is not available.
        FileNotFoundError: If file doesn't exist.

    Example:
        >>> from geosmith.workflows.segy import read_segy_summary
        >>>
        >>> summary = read_segy_summary('seismic.sgy')
        >>> print(f"Traces: {summary.n_traces}, Samples: {summary.n_samples}")
    """
    if not SEGYIO_AVAILABLE:
        raise ImportError(
            "segyio is required for SEG-Y support. "
            "Install with: pip install segyio"
        )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SEG-Y file not found: {path}")

    with segyio.open(
        str(path), mode="r", strict=False, ignore_geometry=True
    ) as f:
        n_traces = f.tracecount
        n_samples = f.samples.size
        dt = float(segyio.tools.dt(f))  # microseconds
        text = segyio.tools.wrap(f.text[0]) if hasattr(f, "text") else ""
        has_ix = hasattr(f, "attributes") and (
            segyio.TraceField.INLINE_3D in f.attributes
            or segyio.TraceField.CROSSLINE_3D in f.attributes
        )

        # Get format from binary header
        format_code = f.bin[segyio.BinField.Format]
        sample_interval = f.bin[segyio.BinField.Interval]

        return SegySummary(
            path=str(path),
            n_traces=int(n_traces),
            n_samples=int(n_samples),
            sample_rate_us=dt,
            text_header=text,
            has_inline_crossline=bool(has_ix),
            format=int(format_code),
            sample_interval_us=float(sample_interval) if sample_interval else dt,
        )


def read_segy_traces(
    path: str | Path,
    trace_indices: Optional[List[int]] = None,
    max_traces: int = 1000,
) -> np.ndarray:
    """Read trace data from SEG-Y file.

    Args:
        path: Path to SEG-Y file.
        trace_indices: Optional list of trace indices to read.
        max_traces: Maximum number of traces to read (for large files).

    Returns:
        Array of shape (n_traces, n_samples) with trace data.

    Raises:
        ImportError: If segyio is not available.

    Example:
        >>> from geosmith.workflows.segy import read_segy_traces
        >>>
        >>> traces = read_segy_traces('seismic.sgy', max_traces=100)
        >>> print(f"Trace data shape: {traces.shape}")
    """
    if not SEGYIO_AVAILABLE:
        raise ImportError(
            "segyio is required for SEG-Y support. "
            "Install with: pip install segyio"
        )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SEG-Y file not found: {path}")

    with segyio.open(
        str(path), mode="r", strict=False, ignore_geometry=True
    ) as f:
        if trace_indices is None:
            # Read all traces (up to max_traces)
            n_read = min(f.tracecount, max_traces)
            trace_indices = list(range(n_read))
        else:
            n_read = len(trace_indices)

        n_samples = f.samples.size
        traces = np.zeros((n_read, n_samples))

        for i, idx in enumerate(trace_indices):
            if idx >= f.tracecount:
                continue
            traces[i, :] = f.trace[idx]

        logger.info(f"Read {n_read} traces from SEG-Y file")

        return traces

