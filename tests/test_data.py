"""Tests for DESI data loading and dataset."""

import numpy as np
import torch
import pytest
from pathlib import Path

from src.utils.data import DESISpectrumDataset, stitch_bands, collate_desi_batch


# Path to downloaded test data
COADD_PATH = Path("data/desi_raw/coadd-sv3-bright-10016.fits")
REDROCK_PATH = Path("data/desi_raw/redrock-sv3-bright-10016.fits")


class TestStitchBands:
    """Test band stitching functionality."""
    
    def test_stitch_no_overlap(self):
        """Test stitching bands with no overlap."""
        wave1 = np.array([3600, 3700, 3800], dtype=np.float32)
        wave2 = np.array([4000, 4100], dtype=np.float32)
        flux1 = np.ones(3, dtype=np.float32)
        flux2 = np.ones(2, dtype=np.float32) * 2
        ivar1 = np.ones(3, dtype=np.float32)
        ivar2 = np.ones(2, dtype=np.float32)
        mask1 = np.zeros(3, dtype=bool)
        mask2 = np.zeros(2, dtype=bool)
        
        result = stitch_bands(
            [wave1, wave2],
            [flux1, flux2],
            [ivar1, ivar2],
            [mask1, mask2],
        )
        
        assert len(result["wavelength"]) == 5
        assert result["flux"][0] == 1.0
        assert result["flux"][-1] == 2.0
    
    def test_stitch_with_overlap(self):
        """Test stitching bands with overlapping wavelengths."""
        wave1 = np.array([3600, 3700, 3800], dtype=np.float32)
        wave2 = np.array([3800, 3900], dtype=np.float32)  # Exactly overlaps at 3800
        flux1 = np.ones(3, dtype=np.float32)
        flux2 = np.ones(2, dtype=np.float32) * 2
        ivar1 = np.ones(3, dtype=np.float32)
        ivar2 = np.ones(2, dtype=np.float32)
        mask1 = np.zeros(3, dtype=bool)
        mask2 = np.zeros(2, dtype=bool)
        
        result = stitch_bands(
            [wave1, wave2],
            [flux1, flux2],
            [ivar1, ivar2],
            [mask1, mask2],
        )
        
        # Should merge overlapping pixels
        assert len(result["wavelength"]) == 4
        # The merged pixel should be weighted average
        assert 1.0 < result["flux"][2] < 2.0


@pytest.mark.skipif(not COADD_PATH.exists(), reason="DESI coadd file not downloaded")
class TestDESISpectrumDataset:
    """Test DESI dataset loading with real data."""
    
    def test_load_real_data(self):
        """Test loading real DESI coadd file."""
        dataset = DESISpectrumDataset(
            coadd_path=COADD_PATH,
            redrock_path=REDROCK_PATH,
        )
        assert len(dataset) > 0
        assert len(dataset) <= 43  # The file has 43 spectra
    
    def test_getitem_shapes(self):
        """Test that spectra have correct tensor shapes."""
        dataset = DESISpectrumDataset(
            coadd_path=COADD_PATH,
            redrock_path=REDROCK_PATH,
            max_spectra=5,
        )
        
        item = dataset[0]
        
        assert isinstance(item["flux"], torch.Tensor)
        assert isinstance(item["ivar"], torch.Tensor)
        assert isinstance(item["mask"], torch.Tensor)
        assert isinstance(item["wavelength"], torch.Tensor)
        assert isinstance(item["z"], torch.Tensor)
        
        # All arrays should have same length
        n_pix = len(item["flux"])
        assert len(item["ivar"]) == n_pix
        assert len(item["mask"]) == n_pix
        assert len(item["wavelength"]) == n_pix
        
        # Redshift should be scalar
        assert item["z"].shape == torch.Size([])
    
    def test_wavelength_range(self):
        """Test that wavelength covers expected DESI range."""
        dataset = DESISpectrumDataset(
            coadd_path=COADD_PATH,
            redrock_path=REDROCK_PATH,
            max_spectra=1,
        )
        
        item = dataset[0]
        wave = item["wavelength"].numpy()
        
        assert wave.min() >= 3600
        assert wave.max() <= 10000
    
    def test_redshift_values(self):
        """Test that redshifts are loaded correctly."""
        dataset = DESISpectrumDataset(
            coadd_path=COADD_PATH,
            redrock_path=REDROCK_PATH,
            max_spectra=10,
        )
        
        for i in range(len(dataset)):
            z = dataset[i]["z"].item()
            assert z >= -0.1  # Allow for small negative redshifts (stars)
            assert z < 10.0   # Reasonable upper bound
    
    def test_collate_batch(self):
        """Test batch collation."""
        dataset = DESISpectrumDataset(
            coadd_path=COADD_PATH,
            redrock_path=REDROCK_PATH,
            max_spectra=5,
        )
        
        batch = [dataset[i] for i in range(3)]
        collated = collate_desi_batch(batch)
        
        assert collated["flux"].shape[0] == 3
        assert collated["z"].shape[0] == 3
        assert collated["flux"].dim() == 2


class TestDataPaths:
    """Test that data files exist."""
    
    def test_coadd_file_exists(self):
        """Check if coadd file was downloaded."""
        assert COADD_PATH.exists(), f"Coadd file not found at {COADD_PATH}"
    
    def test_redrock_file_exists(self):
        """Check if redrock file was downloaded."""
        assert REDROCK_PATH.exists(), f"Redrock file not found at {REDROCK_PATH}"
