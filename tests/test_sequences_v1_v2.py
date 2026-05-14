"""
Tests for V1/V2 tokenizer compatibility in training sequences.

Ensures `tokenize_and_build` works correctly with:
- V1 spectrum tokenizer (encode returns (indices, denorm))
- V2 spectrum tokenizer (encode returns (indices, denorm, skips))
- Both approaches A and B
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch

from src.models.transformer import (
    EOS_TOKEN,
    MASK_TOKEN,
    REDSHIFT_TOKEN_OFFSET,
    SOS_TOKEN,
    SPECTRUM_TOKEN_OFFSET,
)
from src.training.sequences import tokenize_and_build


# ---------------------------------------------------------------------------
# Fake tokenizers that mimic V1 and V2 interfaces
# ---------------------------------------------------------------------------

class FakeSpecTokV1:
    """V1 interface: encode() returns (indices, denorm)."""

    def __init__(self, n_tokens=8, codebook=16):
        self.n_tokens = n_tokens
        self.codebook = codebook

    def encode(self, x):
        B = x.shape[0]
        indices = (
            torch.arange(B).unsqueeze(1) + torch.arange(self.n_tokens).unsqueeze(0)
        ) % self.codebook
        denorm = torch.ones(B)
        return indices, denorm


class FakeSpecTokV2:
    """V2 interface: encode() returns (indices, denorm, skips)."""

    def __init__(self, n_tokens=8, codebook=16):
        self.n_tokens = n_tokens
        self.codebook = codebook

    def encode(self, x):
        B = x.shape[0]
        indices = (
            torch.arange(B).unsqueeze(1) + torch.arange(self.n_tokens).unsqueeze(0)
        ) % self.codebook
        denorm = torch.ones(B)
        skips = [
            torch.randn(B, 96, self.n_tokens * (4 ** i))
            for i in range(4)
        ]
        return indices, denorm, skips


class FakeZTok:
    """Shared redshift tokenizer stub."""

    def encode(self, z):
        if isinstance(z, float):
            return int(abs(z) * 10) % 16
        return (z.abs() * 10).long() % 16

    def fit(self, z):
        pass


def _make_raw_batch(B=2, L=16):
    return {
        "flux": torch.rand(B, L),
        "ivar": torch.rand(B, L) + 0.1,
        "z": torch.linspace(0.0, 1.0, B),
        "mask": torch.zeros(B, L, dtype=torch.bool),
    }


# ---------------------------------------------------------------------------
# Core compatibility tests
# ---------------------------------------------------------------------------

class TestV1TokenizerCompatibility:
    """V1 spectrum tokenizer with sequences.py."""

    def test_v1_encode_returns_two_tuple(self):
        spec = FakeSpecTokV1()
        z = FakeZTok()
        raw = _make_raw_batch()
        enc, dec, tgt, mp, rz_mask = tokenize_and_build(
            raw, spec, z, "a", torch.device("cpu"), encoder_mask_ratio=0.0
        )
        assert enc.shape[0] == 2
        assert dec.shape[0] == 2
        assert tgt.shape[0] == 2

    def test_v1_approach_a_sequence_structure(self):
        spec = FakeSpecTokV1(n_tokens=8)
        z = FakeZTok()
        raw = _make_raw_batch()
        enc, dec, tgt, _, _ = tokenize_and_build(
            raw, spec, z, "a", torch.device("cpu"), encoder_mask_ratio=0.0
        )
        assert enc[:, 0].equal(torch.full_like(enc[:, 0], SOS_TOKEN))
        assert tgt[:, 0].ge(REDSHIFT_TOKEN_OFFSET).all()
        assert enc[:, -1].equal(torch.full_like(enc[:, -1], EOS_TOKEN))
        assert dec[:, 0].equal(torch.full_like(dec[:, 0], SOS_TOKEN))
        assert tgt[-1, -1] == EOS_TOKEN

    def test_v1_approach_b_sequence_structure(self):
        spec = FakeSpecTokV1(n_tokens=8)
        z = FakeZTok()
        raw = _make_raw_batch()
        enc, dec, tgt, _, _ = tokenize_and_build(
            raw, spec, z, "b", torch.device("cpu"), encoder_mask_ratio=0.0
        )
        assert enc[:, 0].equal(torch.full_like(enc[:, 0], SOS_TOKEN))
        assert enc[:, -1].equal(torch.full_like(enc[:, -1], EOS_TOKEN))
        assert tgt[:, 0].ge(REDSHIFT_TOKEN_OFFSET).all()
        assert dec[:, 0].equal(torch.full_like(dec[:, 0], SOS_TOKEN))
        assert tgt.shape[1] == dec.shape[1]
        assert tgt.shape[1] == enc.shape[1]

    def test_v1_with_encoder_masking(self):
        spec = FakeSpecTokV1(n_tokens=8)
        z = FakeZTok()
        raw = _make_raw_batch(B=4)
        enc, dec, tgt, mp, rz_mask = tokenize_and_build(
            raw, spec, z, "a", torch.device("cpu"), encoder_mask_ratio=0.5
        )
        assert mp is not None
        assert 0 < mp.sum() < mp.numel()
        assert dec[:, 0].equal(torch.full_like(dec[:, 0], SOS_TOKEN))


class TestV2TokenizerCompatibility:
    """V2 spectrum tokenizer with sequences.py."""

    def test_v2_encode_returns_three_tuple(self):
        spec = FakeSpecTokV2()
        z = FakeZTok()
        raw = _make_raw_batch()
        enc, dec, tgt, mp, rz_mask = tokenize_and_build(
            raw, spec, z, "a", torch.device("cpu"), encoder_mask_ratio=0.0
        )
        assert enc.shape[0] == 2
        assert dec.shape[0] == 2
        assert tgt.shape[0] == 2

    def test_v2_approach_a_sequence_structure(self):
        spec = FakeSpecTokV2(n_tokens=8)
        z = FakeZTok()
        raw = _make_raw_batch()
        enc, dec, tgt, _, _ = tokenize_and_build(
            raw, spec, z, "a", torch.device("cpu"), encoder_mask_ratio=0.0
        )
        assert enc[:, 0].equal(torch.full_like(enc[:, 0], SOS_TOKEN))
        assert tgt[:, 0].ge(REDSHIFT_TOKEN_OFFSET).all()
        assert enc[:, -1].equal(torch.full_like(enc[:, -1], EOS_TOKEN))
        assert dec[:, 0].equal(torch.full_like(dec[:, 0], SOS_TOKEN))
        assert tgt[-1, -1] == EOS_TOKEN

    def test_v2_approach_b_sequence_structure(self):
        spec = FakeSpecTokV2(n_tokens=8)
        z = FakeZTok()
        raw = _make_raw_batch()
        enc, dec, tgt, _, _ = tokenize_and_build(
            raw, spec, z, "b", torch.device("cpu"), encoder_mask_ratio=0.0
        )
        assert enc[:, 0].equal(torch.full_like(enc[:, 0], SOS_TOKEN))
        assert enc[:, -1].equal(torch.full_like(enc[:, -1], EOS_TOKEN))
        assert tgt[:, 0].ge(REDSHIFT_TOKEN_OFFSET).all()
        assert dec[:, 0].equal(torch.full_like(dec[:, 0], SOS_TOKEN))

    def test_v2_with_encoder_masking(self):
        spec = FakeSpecTokV2(n_tokens=8)
        z = FakeZTok()
        raw = _make_raw_batch(B=4)
        enc, dec, tgt, mp, rz_mask = tokenize_and_build(
            raw, spec, z, "a", torch.device("cpu"), encoder_mask_ratio=0.5
        )
        assert mp is not None
        assert 0 < mp.sum() < mp.numel()
        assert dec[:, 0].equal(torch.full_like(dec[:, 0], SOS_TOKEN))

    def test_v2_skips_not_used_in_sequences(self):
        spec = FakeSpecTokV2()
        z = FakeZTok()
        raw = _make_raw_batch()
        enc, dec, tgt, mp, rz_mask = tokenize_and_build(
            raw, spec, z, "a", torch.device("cpu"), encoder_mask_ratio=0.0
        )
        assert enc.shape[1] == 1 + 1 + spec.n_tokens + 1


class TestV1V2ApproachABoth:
    """Both V1 and V2 must support approaches A and B equally."""

    @pytest.mark.parametrize("spec_cls", [FakeSpecTokV1, FakeSpecTokV2])
    def test_both_approaches_produce_valid_shapes(self, spec_cls):
        spec = spec_cls(n_tokens=8)
        z = FakeZTok()
        raw = _make_raw_batch(B=3)

        for approach in ("a", "b"):
            enc, dec, tgt, mp, rz_mask = tokenize_and_build(
                raw, spec, z, approach, torch.device("cpu"), encoder_mask_ratio=0.3
            )
            assert enc.shape[0] == 3
            assert dec.shape[0] == 3
            assert tgt.shape[0] == 3
            assert enc.shape[1] >= 5
            assert dec.shape[1] >= 5
            assert tgt.shape[1] >= 5
            if approach == "a":
                # enc: [SOS, rz, s..., EOS] = 11, dec: [SOS, rz, s...] = 10, tgt: [rz, s..., EOS] = 10
                assert tgt.shape[1] == dec.shape[1]
                assert tgt.shape[1] == enc.shape[1] - 1
            else:
                # enc: [SOS, s..., EOS] = 10, dec: [SOS, rz, s...] = 10, tgt: [rz, s..., EOS] = 10
                assert tgt.shape[1] == dec.shape[1]
                assert tgt.shape[1] == enc.shape[1]

    @pytest.mark.parametrize("spec_cls", [FakeSpecTokV1, FakeSpecTokV2])
    def test_decoder_does_not_see_encoder_mask(self, spec_cls):
        spec = spec_cls(n_tokens=8)
        z = FakeZTok()
        raw = _make_raw_batch()

        torch.manual_seed(42)
        _, dec_masked, tgt_masked, _, _ = tokenize_and_build(
            raw, spec, z, "a", torch.device("cpu"), encoder_mask_ratio=0.8
        )
        torch.manual_seed(42)
        _, dec_clean, tgt_clean, _, _ = tokenize_and_build(
            raw, spec, z, "a", torch.device("cpu"), encoder_mask_ratio=0.0
        )
        assert torch.equal(dec_masked, dec_clean)
        assert torch.equal(tgt_masked, tgt_clean)

    @pytest.mark.parametrize("spec_cls", [FakeSpecTokV1, FakeSpecTokV2])
    def test_masked_positions_correct_count(self, spec_cls):
        spec = spec_cls(n_tokens=12)
        z = FakeZTok()
        raw = _make_raw_batch(B=5)

        enc, _, _, mp, _ = tokenize_and_build(
            raw, spec, z, "a", torch.device("cpu"), encoder_mask_ratio=0.3
        )
        expected_masked = int(0.3 * 12 * 5)
        assert abs(mp.sum().item() - expected_masked) <= 5


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_batch_size_1_v1(self):
        spec = FakeSpecTokV1(n_tokens=8)
        z = FakeZTok()
        raw = _make_raw_batch(B=1)
        enc, dec, tgt, mp, rz_mask = tokenize_and_build(
            raw, spec, z, "a", torch.device("cpu"), encoder_mask_ratio=0.0
        )
        assert enc.shape[0] == 1
        assert dec.shape[0] == 1

    def test_batch_size_1_v2(self):
        spec = FakeSpecTokV2(n_tokens=8)
        z = FakeZTok()
        raw = _make_raw_batch(B=1)
        enc, dec, tgt, mp, rz_mask = tokenize_and_build(
            raw, spec, z, "a", torch.device("cpu"), encoder_mask_ratio=0.0
        )
        assert enc.shape[0] == 1
        assert dec.shape[0] == 1

    def test_full_mask_ratio_approach_a_v1(self):
        spec = FakeSpecTokV1(n_tokens=8)
        z = FakeZTok()
        raw = _make_raw_batch(B=2)
        enc, _, _, mp, rz_mask = tokenize_and_build(
            raw, spec, z, "a", torch.device("cpu"), encoder_mask_ratio=1.0
        )
        assert mp.all()
        assert rz_mask is not None
        assert rz_mask.all()

    def test_full_mask_ratio_approach_b_v2(self):
        spec = FakeSpecTokV2(n_tokens=8)
        z = FakeZTok()
        raw = _make_raw_batch(B=2)
        enc, _, _, mp, rz_mask = tokenize_and_build(
            raw, spec, z, "b", torch.device("cpu"), encoder_mask_ratio=1.0
        )
        assert mp.all()
        assert rz_mask is None

    def test_v1_and_v2_produce_same_spec_token_count(self):
        raw = _make_raw_batch(B=2)
        z = FakeZTok()
        spec_v1 = FakeSpecTokV1(n_tokens=8)
        spec_v2 = FakeSpecTokV2(n_tokens=8)

        enc1, _, _, _, _ = tokenize_and_build(
            raw, spec_v1, z, "a", torch.device("cpu"), encoder_mask_ratio=0.0
        )
        enc2, _, _, _, _ = tokenize_and_build(
            raw, spec_v2, z, "a", torch.device("cpu"), encoder_mask_ratio=0.0
        )
        assert enc1.shape[1] == enc2.shape[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])