"""
Tests for RedshiftTokenizer
===========================
"""

import torch
import numpy as np
import pytest
from src.tokenizers.redshift import RedshiftTokenizer


class TestRedshiftTokenizer:
    """Test suite for redshift scalar tokenizer."""
    
    def test_initialization(self):
        """Test tokenizer initializes correctly."""
        tok = RedshiftTokenizer(n_levels=256)
        assert tok.n_levels == 256
        assert tok.gaussian_range == 3.0
        assert not tok.is_fitted
    
    def test_fit(self):
        """Test fitting on redshift data."""
        tok = RedshiftTokenizer(n_levels=128)
        z = torch.tensor([0.0, 0.1, 0.2, 0.5, 1.0, 2.0])
        tok.fit(z)
        
        assert tok.is_fitted
        assert tok._min_z == 0.0
        assert tok._max_z == 2.0
        assert len(tok._sorted_z) == 6
    
    def test_encode_not_fitted_raises(self):
        """Test encode raises if not fitted."""
        tok = RedshiftTokenizer()
        with pytest.raises(RuntimeError, match="not fitted"):
            tok.encode(0.5)
    
    def test_decode_not_fitted_raises(self):
        """Test decode raises if not fitted."""
        tok = RedshiftTokenizer()
        with pytest.raises(RuntimeError, match="not fitted"):
            tok.decode(128)
    
    def test_round_trip_uniform(self):
        """Test encode-decode round trip on uniform redshifts."""
        tok = RedshiftTokenizer(n_levels=256)
        z = torch.linspace(0.0, 2.0, 100)
        tok.fit(z)
        
        indices = tok.encode(z)
        z_recon = tok.decode(indices)
        
        # With empirical CDF and small sample, round-trip error is limited by
        # the spacing between sorted training samples (~0.02 for 100 samples).
        # The CDF discretization means error can be up to ~2x sample spacing.
        sample_spacing = (z.max() - z.min()) / (len(z) - 1)
        max_error = (z - z_recon).abs().max()
        assert max_error < sample_spacing * 2.5
    
    def test_round_trip_skewed(self):
        """Test round trip on skewed distribution (many stars, few galaxies)."""
        tok = RedshiftTokenizer(n_levels=256)
        # Simulate DESI-like distribution: 90% stars, 10% galaxies
        # Use more galaxies for better CDF resolution
        z_stars = torch.zeros(90)
        z_galaxies = torch.linspace(0.1, 2.5, 100)
        z = torch.cat([z_stars, z_galaxies])
        tok.fit(z)
        
        indices = tok.encode(z)
        z_recon = tok.decode(indices)
        
        # Stars should reconstruct near 0
        star_error = (z_recon[:90] - 0.0).abs().max()
        assert star_error < 0.05
        
        # Galaxies should have moderate error (empirical CDF with small sample)
        gal_error = (z_recon[90:] - z_galaxies).abs()
        assert gal_error.max() < 0.15  # Absolute error bound
    
    def test_monotonicity(self):
        """Test that higher redshift maps to equal or higher token index."""
        tok = RedshiftTokenizer(n_levels=256)
        z = torch.linspace(0.0, 3.0, 500)
        tok.fit(z)
        
        indices = tok.encode(z)
        
        # Check monotonicity (non-decreasing)
        diffs = indices[1:] - indices[:-1]
        assert (diffs >= 0).all()
    
    def test_all_bins_used_uniform(self):
        """Test that all quantization bins are used for uniform distribution."""
        tok = RedshiftTokenizer(n_levels=64)
        z = torch.linspace(0.0, 2.0, 1000)
        tok.fit(z)
        
        indices = tok.encode(z)
        unique = torch.unique(indices)
        
        # Should use most bins (allow some edge effects)
        assert len(unique) >= tok.n_levels * 0.8
    
    def test_boundary_conditions(self):
        """Test encoding extreme values."""
        tok = RedshiftTokenizer(n_levels=256)
        z = torch.tensor([0.0, 0.5, 1.0])
        tok.fit(z)
        
        # Values outside range should map to boundary bins
        idx_low = tok.encode(torch.tensor(-1.0))
        idx_high = tok.encode(torch.tensor(2.0))
        
        assert idx_low.item() >= 0
        assert idx_high.item() < tok.n_levels
    
    def test_batch_consistency(self):
        """Test batch and single encoding give same results."""
        tok = RedshiftTokenizer(n_levels=128)
        z = torch.tensor([0.0, 0.1, 0.5, 1.0, 2.0])
        tok.fit(z)
        
        batch_indices = tok.encode(z)
        single_indices = torch.stack([tok.encode(zi) for zi in z])
        
        assert torch.equal(batch_indices, single_indices.squeeze())
    
    def test_scalar_input(self):
        """Test scalar input works."""
        tok = RedshiftTokenizer(n_levels=64)
        z = torch.linspace(0.0, 1.0, 50)
        tok.fit(z)
        
        idx = tok.encode(0.5)
        assert isinstance(idx, torch.Tensor)
        assert idx.numel() == 1
        
        z_recon = tok.decode(idx)
        assert z_recon.numel() == 1
    
    def test_numpy_input(self):
        """Test numpy array input works."""
        tok = RedshiftTokenizer(n_levels=64)
        z_np = np.array([0.0, 0.1, 0.5, 1.0])
        tok.fit(z_np)
        
        indices = tok.encode(z_np)
        assert isinstance(indices, torch.Tensor)
        assert len(indices) == 4
    
    def test_cdf_monotonic(self):
        """Test CDF transform is monotonic."""
        tok = RedshiftTokenizer()
        z = torch.tensor([0.0, 0.1, 0.2, 0.5, 1.0])
        tok.fit(z)
        
        cdf = tok._cdf(z)
        diffs = cdf[1:] - cdf[:-1]
        assert (diffs >= 0).all()
    
    def test_gaussian_transform(self):
        """Test CDF <-> Gaussian transforms are inverses."""
        tok = RedshiftTokenizer()
        # Use uniform CDF values
        cdf = torch.tensor([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
        
        gaussian = tok.cdf_to_gaussian(cdf)
        cdf_back = tok.gaussian_to_cdf(gaussian)
        
        assert torch.allclose(cdf, cdf_back, atol=1e-5)
    
    def test_desi_real_data(self):
        """Test on actual DESI redshift distribution."""
        from pathlib import Path
        from src.utils.data import DESISpectrumDataset
        
        # Load real redshifts
        files = sorted(Path('data/desi_raw').glob('coadd-*.fits'))
        all_z = []
        for f in files:
            rr = f.parent / f.name.replace('coadd-', 'redrock-')
            ds = DESISpectrumDataset(coadd_path=f, redrock_path=rr,
                                     require_good_zwarn=False, require_nonzero_flux=True)
            for i in range(len(ds)):
                all_z.append(ds[i]['z'].item())
        
        all_z = torch.tensor(all_z)
        
        tok = RedshiftTokenizer(n_levels=256)
        tok.fit(all_z)
        
        # Encode all
        indices = tok.encode(all_z)
        z_recon = tok.decode(indices)
        
        # Check utilization
        unique = torch.unique(indices)
        utilization = len(unique) / tok.n_levels
        print(f"Token utilization: {utilization:.2%} ({len(unique)}/{tok.n_levels})")
        
        # At least 10% utilization on real data
        assert utilization > 0.1
        
        # Check reconstruction accuracy
        mae = (all_z - z_recon).abs().mean()
        print(f"Mean absolute error: {mae:.6f}")
        
        # For stars (z~0), should be very accurate
        star_mask = all_z < 0.01
        if star_mask.sum() > 0:
            star_mae = (all_z[star_mask] - z_recon[star_mask]).abs().mean()
            print(f"Star MAE: {star_mae:.6f}")
            assert star_mae < 0.01
    
    def test_get_bin_edges(self):
        """Test bin edge computation."""
        tok = RedshiftTokenizer(n_levels=16)
        z = torch.linspace(0.0, 2.0, 100)
        tok.fit(z)
        
        edges = tok.get_bin_edges()
        assert len(edges) == tok.n_levels + 1
        assert edges[0] <= edges[-1]  # Monotonic
