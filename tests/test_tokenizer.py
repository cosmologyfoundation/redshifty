"""Tests for spectrum tokenizer."""

import torch
import pytest
import numpy as np

from src.tokenizers.spectrum import (
    SpectrumTokenizer,
    ConvNeXtBlock1D,
    LookUpFreeQuantizer,
    LATENT_GRID_SIZE,
)


class TestConvNeXtBlock1D:
    """Test ConvNeXt block."""
    
    def test_shape_preserved(self):
        """Block should preserve (B, C, L) shape."""
        block = ConvNeXtBlock1D(dim=64)
        x = torch.randn(2, 64, 128)
        out = block(x)
        assert out.shape == x.shape
    
    def test_residual_connection(self):
        """Output should be close to input for small weights."""
        block = ConvNeXtBlock1D(dim=32)
        # Zero out weights so block is near identity
        with torch.no_grad():
            for p in block.parameters():
                p.fill_(0)
        
        x = torch.randn(1, 32, 64)
        out = block(x)
        # Should be close to input (just residual)
        assert torch.allclose(out, x, atol=1e-5)


class TestLookUpFreeQuantizer:
    """Test LFQ quantizer."""
    
    def test_quantize_shape(self):
        """Quantized output should have same shape as input."""
        lfq = LookUpFreeQuantizer(dim=8, codebook_size=256)
        z = torch.randn(2, 8, 16)
        z_q, loss, indices = lfq(z)
        
        assert z_q.shape == z.shape
        assert indices.shape == (2, 16)
        assert loss.item() >= 0
    
    def test_encode_decode_values(self):
        """Decode should produce {-1, +1} values."""
        lfq = LookUpFreeQuantizer(dim=8, codebook_size=1024)
        z = torch.randn(2, 8, 16)
        
        indices = lfq.encode(z)
        z_q = lfq.decode(indices)
        
        # Decoded values should be exactly -1 or +1
        assert torch.all((z_q == -1) | (z_q == 1))
        assert z_q.shape == z.shape
    
    def test_codebook_size(self):
        """Indices should be in valid range."""
        lfq = LookUpFreeQuantizer(dim=4, codebook_size=256)
        z = torch.randn(2, 4, 16)
        
        indices = lfq.encode(z)
        assert indices.min() >= 0
        assert indices.max() < 256


class TestSpectrumTokenizer:
    """Test full tokenizer."""
    
    def test_forward_shape(self):
        """Forward pass should return correct shapes."""
        model = SpectrumTokenizer()
        x = torch.randn(2, 2, 7781)
        
        recon, loss, indices = model(x)
        
        assert recon.shape == (2, 2, LATENT_GRID_SIZE)
        assert indices.ndim == 2  # (B, n_tokens) - each position gets one integer code
        assert "total" in loss
        assert "recon" in loss
        assert "quant" in loss
    
    def test_encode_decode_shape(self):
        """Encode -> decode roundtrip."""
        model = SpectrumTokenizer()
        x = torch.randn(1, 2, 7781)
        
        indices, denorm = model.encode(x)
        recon = model.decode(indices, denorm)
        
        assert recon.shape == (1, 2, LATENT_GRID_SIZE)
    
    def test_encode_decode_consistency(self):
        """Forward encode should match standalone encode (in eval mode)."""
        model = SpectrumTokenizer()
        model.eval()
        x = torch.randn(1, 2, 7781)
        
        with torch.no_grad():
            _, _, indices_fwd = model(x)
            indices_enc, _ = model.encode(x)
        
        assert torch.equal(indices_fwd, indices_enc)
    
    def test_different_batch_sizes(self):
        """Should work with different batch sizes."""
        model = SpectrumTokenizer()
        
        for bs in [1, 4, 8]:
            x = torch.randn(bs, 2, 7781)
            recon, loss, indices = model(x)
            assert recon.shape[0] == bs
            assert indices.shape[0] == bs
    
    def test_different_input_lengths(self):
        """Should interpolate different input lengths."""
        model = SpectrumTokenizer()
        
        for length in [7000, 7781, 8000]:
            x = torch.randn(1, 2, length)
            recon, loss, indices = model(x)
            assert recon.shape == (1, 2, LATENT_GRID_SIZE)
    
    def test_reconstruction_loss_decreases_with_training(self):
        """Quick check that model can overfit a single sample."""
        model = SpectrumTokenizer()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        x = torch.randn(1, 2, 7781)
        
        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            recon, loss, _ = model(x)
            loss["total"].backward()
            optimizer.step()
            losses.append(loss["recon"].item())
        
        # Loss should generally decrease
        assert losses[-1] < losses[0]
    
    def test_decoder_not_constant(self):
        """Decoder should output non-constant values (regression test for #1)."""
        model = SpectrumTokenizer()
        x = torch.randn(1, 2, 8704)
        
        model.eval()
        with torch.no_grad():
            recon, _, _ = model(x)
        
        # Reconstruction should have variance > 0
        assert recon.std() > 0.01, f"Decoder output is constant: std={recon.std():.6f}"
    
    def test_parameter_count(self):
        """Model should have reasonable parameter count."""
        model = SpectrumTokenizer()
        n_params = sum(p.numel() for p in model.parameters())
        
        # Should be in the 10M-50M range for smoke test model
        assert 5_000_000 < n_params < 50_000_000
        print(f"\nTokenizer parameters: {n_params:,}")
