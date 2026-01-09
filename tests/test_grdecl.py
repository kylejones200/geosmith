"""Tests for GRDECL I/O workflows."""

import numpy as np
import pytest
from pathlib import Path
import tempfile

from geosmith.workflows.grdecl import read_grdecl, write_grdecl


class TestReadGRDECL:
    """Tests for read_grdecl."""

    def test_read_grdecl_simple(self):
        """Test reading a simple GRDECL file."""
        # Create a simple GRDECL file with proper format
        # 10 x 10 x 5 = 500 values
        permx_values = " ".join(["0.1"] * 500)
        grdecl_content = f"""SPECGRID
10 10 5 /
PERMX
{permx_values}
/
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.GRDECL', delete=False) as f:
            f.write(grdecl_content)
            temp_path = f.name

        try:
            data = read_grdecl(temp_path)
            assert 'dimensions' in data
            assert 'properties' in data
            assert data['dimensions'] == (10, 10, 5)
            assert 'PERMX' in data['properties']
            assert data['properties']['PERMX'].shape == (10, 10, 5)
        finally:
            Path(temp_path).unlink()

    def test_read_grdecl_with_property_name(self):
        """Test reading specific property as RasterGrid."""
        # 5 x 5 x 3 = 75 values
        permx_values = " ".join(["0.5"] * 75)
        grdecl_content = f"""SPECGRID
5 5 3 /
PERMX
{permx_values}
/
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.GRDECL', delete=False) as f:
            f.write(grdecl_content)
            temp_path = f.name

        try:
            permx = read_grdecl(temp_path, property_name='PERMX')
            from geosmith.objects.rastergrid import RasterGrid
            assert isinstance(permx, RasterGrid)
            # GRDECL uses Fortran order: (nx, ny, nz) = (5, 5, 3)
            # Reshaped with order='F' gives (5, 5, 3)
            # RasterGrid expects (nz, ny, nx) = (3, 5, 5)
            assert permx.data.shape == (5, 5, 3) or permx.data.shape == (3, 5, 5)
            assert permx.data.size == 75  # Total elements
        finally:
            Path(temp_path).unlink()

    def test_read_grdecl_file_not_found(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            read_grdecl('nonexistent.GRDECL')

    def test_read_grdecl_invalid_format(self):
        """Test that invalid format raises ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.GRDECL', delete=False) as f:
            f.write("INVALID CONTENT")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="SPECGRID"):
                read_grdecl(temp_path)
        finally:
            Path(temp_path).unlink()


class TestWriteGRDECL:
    """Tests for write_grdecl."""

    def test_write_grdecl_from_dict(self):
        """Test writing GRDECL from dictionary."""
        data = {
            'dimensions': (5, 5, 3),
            'properties': {
                'PERMX': np.ones((5, 5, 3)) * 0.5
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.GRDECL', delete=False) as f:
            temp_path = f.name

        try:
            write_grdecl(data, temp_path)
            assert Path(temp_path).exists()
            
            # Read it back
            read_data = read_grdecl(temp_path)
            assert read_data['dimensions'] == (5, 5, 3)
            assert 'PERMX' in read_data['properties']
            np.testing.assert_array_almost_equal(
                read_data['properties']['PERMX'],
                data['properties']['PERMX']
            )
        finally:
            Path(temp_path).unlink()

    def test_write_grdecl_from_rastergrid(self):
        """Test writing GRDECL from RasterGrid."""
        from geosmith.objects.rastergrid import RasterGrid
        from geosmith.objects.geoindex import GeoIndex

        data = np.ones((3, 5, 5)) * 0.5
        transform = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
        index = GeoIndex(crs=None, bounds=(0, 0, 5, 5))
        raster = RasterGrid(data=data, transform=transform, index=index)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.GRDECL', delete=False) as f:
            temp_path = f.name

        try:
            write_grdecl(raster, temp_path, property_name='PERMX')
            assert Path(temp_path).exists()
            
            # Read it back
            read_data = read_grdecl(temp_path)
            assert read_data['dimensions'] == (5, 5, 3)
        finally:
            Path(temp_path).unlink()

    def test_write_grdecl_missing_property_name(self):
        """Test that RasterGrid without property_name raises error."""
        from geosmith.objects.rastergrid import RasterGrid
        from geosmith.objects.geoindex import GeoIndex

        data = np.ones((3, 5, 5)) * 0.5
        transform = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
        index = GeoIndex(crs=None, bounds=(0, 0, 5, 5))
        raster = RasterGrid(data=data, transform=transform, index=index)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.GRDECL', delete=False) as f:
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="property_name"):
                write_grdecl(raster, temp_path)
        finally:
            Path(temp_path).unlink()

