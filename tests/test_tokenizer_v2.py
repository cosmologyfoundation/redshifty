"""Tests for spectrum tokenizer V2."""

import torch
import pytest
import math

from src.tokenizers.spectrum_v2 import (
    SpectrumTokenizerV2,
    ConvNeXtBlock1D,
    LookUpFreeQuantizerV2,
    CrossAttention,
    EntropyLoss,
    TopHatSmoothing,
    LATENT_GRID_SIZE,
)


class TestConvNeXtBlock1D:
    def test_shape_preserved(self):
        block = ConvNeXtBlock1D(dim=64)
        x = torch.randn(2, 64, 128)
        out = block(x)
        assert out.shape == x.shape

    def test_residual_connection(self):
        block = ConvNeXtBlock1D(dim=32)
        with torch.no_grad():
            for p in block.parameters():
                p.fill_(0)
        x = torch.randn(1, 32, 64)
        out = block(x)
        assert torch.allclose(out, x, atol=1e-5)


class TestCrossAttention:
    def test_shape_preserved(self):
        attn = CrossAttention(dim=128)
        query = torch.randn(2, 128, 64)
        key_value = torch.randn(2, 128, 64)
        out = attn(query, key_value)
        assert out.shape == query.shape

    def test_different_length_key_value(self):
        """Cross-attention should handle key_value with different sequence length."""
        attn = CrossAttention(dim=128)
        query = torch.randn(2, 128, 32)
        key_value = torch.randn(2, 128, 64)
        out = attn(query, key_value)
        assert out.shape == query.shape

    def test_same_dim_different_length(self):
        """Cross-attention should work when query and key_value are same dim, different length."""
        attn = CrossAttention(dim=192)
        query = torch.randn(2, 192, 16)
        key_value = torch.randn(2, 192, 48)
        out = attn(query, key_value)
        assert out.shape == query.shape


class TestEntropyLoss:
    def test_uniform_codebook_zero_loss(self):
        """Uniform distribution over codes should give zero entropy loss."""
        entropy_fn = EntropyLoss(codebook_size=1024, entropy_weight=1.0)
        indices = torch.arange(1024).unsqueeze(0).expand(100, -1)
        loss = entropy_fn(indices)
        assert loss.item() < 1e-3

    def test_peaked_codebook_positive_loss(self):
        """Peaked distribution should give positive entropy loss."""
        entropy_fn = EntropyLoss(codebook_size=1024, entropy_weight=1.0)
        indices = torch.zeros(100, 50, dtype=torch.long)
        loss = entropy_fn(indices)
        assert loss.item() > 0.5

    def test_loss_depends_on_codebook_size(self):
        """Larger codebook with same usage should give different loss."""
        entropy_fn_small = EntropyLoss(codebook_size=256, entropy_weight=0.1)
        entropy_fn_large = EntropyLoss(codebook_size=1024, entropy_weight=0.1)
        indices = torch.randint(0, 256, (10, 100))
        loss_small = entropy_fn_small(indices)
        loss_large = entropy_fn_large(indices)
        assert loss_large > loss_small


class TestTopHatSmoothing:
    def test_output_shape(self):
        smoothing = TopHatSmoothing(channels=2)
        x = torch.randn(2, 2, 100)
        out = smoothing(x)
        assert out.shape == x.shape

    def test_reduces_high_frequency(self):
        """Smoothing should reduce amplitude of high-frequency variations."""
        smoothing = TopHatSmoothing(channels=1)
        t = torch.linspace(0, 4 * math.pi, 200).unsqueeze(0).unsqueeze(0)
        x = torch.sin(t)
        out = smoothing(x)
        assert out.std() < x.std()

    def test_padding_preserves_length(self):
        smoothing = TopHatSmoothing(channels=1)
        x = torch.randn(1, 1, 100)
        out = smoothing(x)
        assert out.shape[-1] == x.shape[-1]


class TestLookUpFreeQuantizerV2:
    def test_quantize_shape(self):
        lfq = LookUpFreeQuantizerV2(dim=8, codebook_size=256)
        z = torch.randn(2, 8, 16)
        z_q, commit, entropy, indices = lfq(z)
        assert z_q.shape == z.shape
        assert indices.shape == (2, 16)
        assert commit.item() >= 0
        assert entropy.item() >= 0

    def test_commitment_loss_positive(self):
        lfq = LookUpFreeQuantizerV2(dim=8, codebook_size=256)
        z = torch.randn(2, 8, 16)
        _, commit, _, _ = lfq(z)
        assert commit.item() >= 0

    def test_entropy_loss_positive(self):
        lfq = LookUpFreeQuantizerV2(dim=8, codebook_size=256)
        z = torch.randn(2, 8, 16)
        _, _, entropy, _ = lfq(z)
        assert entropy.item() >= 0

    def test_encode_decode_values(self):
        lfq = LookUpFreeQuantizerV2(dim=8, codebook_size=256)
        z = torch.randn(2, 8, 16)
        indices = lfq.encode(z)
        z_q = lfq.decode(indices)
        assert torch.all((z_q == -1) | (z_q == 1))
        assert z_q.shape == z.shape

    def test_codebook_size(self):
        lfq = LookUpFreeQuantizerV2(dim=4, codebook_size=256)
        z = torch.randn(2, 4, 16)
        indices = lfq.encode(z)
        assert indices.min() >= 0
        assert indices.max() < 256


class TestSpectrumTokenizerV2:
    def test_forward_shape(self):
        model = SpectrumTokenizerV2()
        x = torch.randn(2, 2, 7781)
        recon, loss, indices = model(x)
        assert recon.shape == (2, 2, LATENT_GRID_SIZE)
        assert indices.ndim == 2
        assert "total" in loss
        assert "recon" in loss
        assert "quant" in loss
        assert "commit" in loss
        assert "entropy" in loss

    def test_encode_decode_roundtrip(self):
        model = SpectrumTokenizerV2(use_skip_connections=True, use_cross_attention=True)
        x = torch.randn(2, 2, 7781)
        indices, denorm, skips = model.encode(x)
        recon = model.decode(indices, denorm, skips)
        assert recon.shape == (2, 2, LATENT_GRID_SIZE)
        assert len(skips) == 4

    def test_encode_skips_have_correct_shapes(self):
        model = SpectrumTokenizerV2()
        x = torch.randn(1, 2, 7781)
        _, _, skips = model.encode(x)
        assert len(skips) == 4
        assert skips[0].shape[1] == 96   # Stage 1: 96 channels
        assert skips[1].shape[1] == 192  # Stage 2: 192 channels
        assert skips[2].shape[1] == 384  # Stage 3: 384 channels
        assert skips[3].shape[1] == 512  # Stage 4: 512 channels

    def test_encode_decode_consistency(self):
        """Forward pass encode should match standalone encode (eval mode)."""
        model = SpectrumTokenizerV2()
        model.eval()
        x = torch.randn(1, 2, 7781)
        with torch.no_grad():
            _, _, indices_fwd = model(x)
            indices_enc, _, _ = model.encode(x)
        assert torch.equal(indices_fwd, indices_enc)

    def test_all_configurations(self):
        """Test all combinations of feature flags."""
        for use_tophat in [True, False]:
            for use_skip in [True, False]:
                for use_ca in [True, False]:
                    if not use_skip and use_ca:
                        continue
                    model = SpectrumTokenizerV2(
                        use_tophat=use_tophat,
                        use_skip_connections=use_skip,
                        use_cross_attention=use_ca,
                    )
                    x = torch.randn(2, 2, 7781)
                    recon, loss, indices = model(x)
                    assert recon.shape == (2, 2, LATENT_GRID_SIZE)
                    assert indices.shape[1] == 272
                    assert loss["recon"].item() >= 0
                    assert loss["quant"].item() >= 0

    def test_different_batch_sizes(self):
        model = SpectrumTokenizerV2()
        for bs in [1, 4, 8]:
            x = torch.randn(bs, 2, 7781)
            recon, loss, indices = model(x)
            assert recon.shape[0] == bs
            assert indices.shape[0] == bs

    def test_different_input_lengths(self):
        model = SpectrumTokenizerV2()
        for length in [7000, 7781, 8000]:
            x = torch.randn(2, 2, length)
            recon, loss, indices = model(x)
            assert recon.shape == (2, 2, LATENT_GRID_SIZE)

    def test_reconstruction_loss_decreases_with_training(self):
        """Model should be able to overfit a single sample."""
        model = SpectrumTokenizerV2()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randn(1, 2, 7781)
        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            recon, loss, _ = model(x)
            loss["total"].backward()
            optimizer.step()
            losses.append(loss["recon"].item())
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"

    def test_entropy_loss_is_bounded(self):
        """Entropy loss should stay bounded and positive throughout training."""
        model = SpectrumTokenizerV2(entropy_weight=0.5)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randn(8, 2, 7781)
        entropy_losses = []
        for _ in range(20):
            optimizer.zero_grad()
            _, loss, _ = model(x)
            loss["total"].backward()
            optimizer.step()
            entropy_losses.append(loss["entropy"].item())
        assert all(0 <= e <= 1.0 for e in entropy_losses), \
            f"Entropy loss out of bounds: min={min(entropy_losses):.4f}, max={max(entropy_losses):.4f}"

    def test_commitment_weight_controls_z_magnitude(self):
        """Higher commitment weight should pull z values toward +/-1."""
        model_low = SpectrumTokenizerV2(commitment_weight=0.01)
        model_high = SpectrumTokenizerV2(commitment_weight=1.0)
        x = torch.randn(2, 2, 7781)
        with torch.no_grad():
            recon_low, loss_low, _ = model_low(x)
            recon_high, loss_high, _ = model_high(x)
        assert loss_high["commit"].item() > loss_low["commit"].item()

    def test_tophat_smooths_input(self):
        model_with = SpectrumTokenizerV2(use_tophat=True)
        model_without = SpectrumTokenizerV2(use_tophat=False)
        x = torch.randn(2, 2, 7781)
        with torch.no_grad():
            # Without tophat: raw input variance
            flux_var_without = x[:, 0, :].var().item()
            # With tophat: smoothed
            x_smoothed = model_with.tophat(model_with.normalize(x)[0])
            flux_var_with = x_smoothed[:, 0, :].var().item()
        assert flux_var_with < flux_var_without

    def test_decoder_not_constant(self):
        """Decoder should output non-constant values (regression test)."""
        model = SpectrumTokenizerV2()
        x = torch.randn(1, 2, 8704)
        model.eval()
        with torch.no_grad():
            recon, _, _ = model(x)
        assert recon.std() > 0.01, f"Decoder output is constant: std={recon.std():.6f}"

    def test_parameter_count(self):
        model = SpectrumTokenizerV2()
        n_params = sum(p.numel() for p in model.parameters())
        assert 20_000_000 < n_params < 50_000_000, \
            f"Parameter count {n_params:,} outside expected range"
        print(f"\nTokenizerV2 parameters: {n_params:,}")

    def test_decode_without_cross_attention_interpolates_skips(self):
        """decode() path without cross-attention should interpolate skip lengths."""
        model = SpectrumTokenizerV2(use_skip_connections=True, use_cross_attention=False)
        x = torch.randn(2, 2, 7781)
        indices, denorm, skips = model.encode(x)
        recon = model.decode(indices, denorm, skips)
        assert recon.shape == (2, 2, LATENT_GRID_SIZE)

    def test_full_training_step(self):
        """Test a single full training step (forward + backward + optimizer step)."""
        model = SpectrumTokenizerV2()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.randn(4, 2, 7781)

        optimizer.zero_grad()
        recon, loss, indices = model(x)
        loss["total"].backward()
        optimizer.step()

        assert all(not torch.isnan(v) for v in loss.values())
        assert all(not torch.isinf(v) for v in loss.values())
        assert indices.shape == (4, 272)

    def test_quant_loss_split_correct(self):
        """commit loss + entropy loss should sum to total quant loss."""
        model = SpectrumTokenizerV2(commitment_weight=0.05, entropy_weight=0.1)
        x = torch.randn(2, 2, 7781)
        _, loss, _ = model(x)
        assert pytest.approx(loss["quant"].item(), abs=1e-6) == loss["commit"].item() + loss["entropy"].item()

    def test_n_tokens_matches_encoder_output(self):
        """N_TOKENS should equal the spatial dimension after encoder stages."""
        model = SpectrumTokenizerV2()
        x = torch.randn(1, 2, 7781)
        with torch.no_grad():
            indices, _, _ = model.encode(x)
        from src.tokenizers.spectrum_v2 import N_TOKENS
        assert indices.shape[1] == N_TOKENS, \
            f"N_TOKENS={N_TOKENS} but actual indices have {indices.shape[1]} tokens"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])