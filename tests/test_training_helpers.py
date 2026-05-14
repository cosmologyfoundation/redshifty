"""
Tests for src/training/ helpers added in Phase 10.

Covers:
- Encoder masking inside `tokenize_and_build` (sequences.py)
- `compute_masked_metrics` (utils.py)
- `split_records_by_healpix` (data_split.py)
- `redshift_weight` behavior in `SpectrumTransformer.forward`
- AR eval glue (`evaluate_ar` in eval.py)
- `init_wandb` mode forcing (wandb_util.py)

Everything runs on CPU with tiny shapes so the full suite stays fast.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

# Ensure repo root is on sys.path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.models.transformer import (
    EOS_TOKEN,
    MASK_TOKEN,
    REDSHIFT_TOKEN_OFFSET,
    SOS_TOKEN,
    SPECTRUM_TOKEN_OFFSET,
    TOTAL_VOCAB_SIZE,
    SpectrumTransformer,
)
from src.training.data_split import split_records_by_healpix
from src.training.eval import evaluate_ar
from src.training.sequences import tokenize_and_build
from src.training.utils import compute_masked_metrics, compute_masked_redshift_acc
from src.training.utils import compute_masked_auc, compute_masked_r2, compute_all_auc, compute_all_r2
from src.training.wandb_util import init_wandb, log_model_artifact


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class FakeSpecTok:
    """Returns a deterministic small integer index per spectrum position."""

    def __init__(self, n_tokens=8, codebook=16):
        self.n_tokens = n_tokens
        self.codebook = codebook

    def encode(self, x):
        # x: (B, 2, L). We ignore L and just hash by batch index.
        B = x.shape[0]
        indices = (torch.arange(B).unsqueeze(1) + torch.arange(self.n_tokens).unsqueeze(0)) % self.codebook
        denorm = torch.ones(B)
        return indices, denorm


class FakeZTok:
    """Maps any float z -> deterministic bin."""

    def encode(self, z):
        return int(abs(z) * 10) % 16


def _make_raw_batch(B=2, L=16):
    return {
        "flux": torch.rand(B, L),
        "ivar": torch.rand(B, L) + 0.1,
        "z": torch.linspace(0.0, 1.0, B),
        "mask": torch.zeros(B, L, dtype=torch.bool),
    }


# ---------------------------------------------------------------------------
# Encoder masking — tests 1-5
# ---------------------------------------------------------------------------

class TestEncoderMasking:
    def test_zero_ratio_no_op(self):
        spec, z = FakeSpecTok(), FakeZTok()
        raw = _make_raw_batch()
        enc0, dec0, tgt0, mp0, rz0 = tokenize_and_build(
            raw, spec, z, "a", torch.device("cpu"), encoder_mask_ratio=0.0
        )
        assert mp0 is None
        assert rz0 is None
        # No MASK token should appear in the encoder for this batch
        assert (enc0 == MASK_TOKEN).sum().item() == 0

    def test_full_ratio_all_spectrum_positions_masked(self):
        spec, z = FakeSpecTok(n_tokens=8), FakeZTok()
        raw = _make_raw_batch(B=2)
        enc, dec, tgt, mp, rz_mask = tokenize_and_build(
            raw, spec, z, "a", torch.device("cpu"), encoder_mask_ratio=1.0
        )
        assert mp is not None
        assert mp.dtype == torch.bool
        assert mp.shape == (2, 8)
        assert mp.all()
        # Encoder layout for A: [SOS, redshift, s1..sN, EOS]
        # Spectrum positions are 2..2+8 = 2..10
        spec_slice = enc[:, 2:2 + 8]
        assert (spec_slice == MASK_TOKEN).all()
        # SOS / redshift / EOS are not MASK (rz is also masked at ratio=1.0)
        assert (enc[:, 0] == SOS_TOKEN).all()
        assert (enc[:, 1] == MASK_TOKEN).all()  # rz masked at ratio=1.0
        assert (enc[:, -1] == EOS_TOKEN).all()
        assert rz_mask is not None
        assert rz_mask.all()

    def test_masking_does_not_touch_decoder_or_target(self):
        spec, z = FakeSpecTok(), FakeZTok()
        raw = _make_raw_batch()
        torch.manual_seed(0)
        _, dec_a, tgt_a, _, _ = tokenize_and_build(
            raw, spec, z, "a", torch.device("cpu"), encoder_mask_ratio=0.0
        )
        torch.manual_seed(0)
        _, dec_b, tgt_b, _, _ = tokenize_and_build(
            raw, spec, z, "a", torch.device("cpu"), encoder_mask_ratio=0.5
        )
        assert torch.equal(dec_a, dec_b)
        assert torch.equal(tgt_a, tgt_b)

    def test_masked_positions_shape_and_dtype(self):
        spec, z = FakeSpecTok(n_tokens=12), FakeZTok()
        raw = _make_raw_batch(B=3)
        enc, _, _, mp, _ = tokenize_and_build(
            raw, spec, z, "b", torch.device("cpu"), encoder_mask_ratio=0.3
        )
        assert mp.shape == (3, 12)
        assert mp.dtype == torch.bool

    def test_approach_b_does_not_mask_redshift(self):
        # Approach B has no redshift token in the encoder; the mask
        # should still only apply to spectrum positions (positions
        # 1..1+T in B).
        spec, z = FakeSpecTok(n_tokens=8), FakeZTok()
        raw = _make_raw_batch(B=2)
        enc, _, _, mp, rz_mask = tokenize_and_build(
            raw, spec, z, "b", torch.device("cpu"), encoder_mask_ratio=1.0
        )
        # B encoder: [SOS, s1..s8, EOS]
        assert (enc[:, 0] == SOS_TOKEN).all()
        assert (enc[:, 1:1 + 8] == MASK_TOKEN).all()
        assert (enc[:, -1] == EOS_TOKEN).all()
        assert rz_mask is None


# ---------------------------------------------------------------------------
# compute_masked_metrics — tests 6-9
# ---------------------------------------------------------------------------

class TestMaskedMetrics:
    def _craft(self, B=2, T=8, V=64):
        logits = torch.zeros(B, T + 2, V)
        # target = [redshift, s1..sT, EOS]
        target = torch.zeros(B, T + 2, dtype=torch.long)
        target[:, 0] = 50  # any non-MASK
        target[:, 1:1 + T] = torch.arange(T).unsqueeze(0).expand(B, -1) + 10
        target[:, -1] = EOS_TOKEN
        # mask positions are aligned with spec tokens (T entries)
        masked = torch.zeros(B, T, dtype=torch.bool)
        return logits, target, masked

    def test_all_correct_when_logits_match(self):
        logits, target, mp = self._craft()
        # Set logits to be max at target positions
        for b in range(target.shape[0]):
            for t in range(target.shape[1]):
                logits[b, t, target[b, t]] = 1.0
        mp.fill_(True)
        out = compute_masked_metrics(logits, target, mp)
        assert out["n_masked"] > 0
        assert out["masked_spec_acc"] == pytest.approx(1.0)

    def test_all_wrong(self):
        logits, target, mp = self._craft(V=64)
        # All argmax = 0, target != 0 at masked positions
        mp.fill_(True)
        out = compute_masked_metrics(logits, target, mp)
        assert out["masked_spec_acc"] == pytest.approx(0.0)

    def test_offset_correctness(self):
        # Mask only position 0 of the spectrum slice (= target index 1)
        logits, target, mp = self._craft(B=1, T=4)
        # Make argmax correct at target index 1 (which is spec pos 0)
        logits[0, 1, target[0, 1]] = 1.0
        mp[0, 0] = True
        # Other spec positions argmax to 0 (wrong)
        out = compute_masked_metrics(logits, target, mp)
        assert out["n_masked"] == 1
        assert out["masked_spec_acc"] == pytest.approx(1.0)

    def test_zero_mask_returns_nan(self):
        logits, target, mp = self._craft()
        out = compute_masked_metrics(logits, target, mp)
        assert out["n_masked"] == 0
        assert out["masked_spec_acc"] != out["masked_spec_acc"]  # NaN check


# ---------------------------------------------------------------------------
# Healpix split — tests 10-12
# ---------------------------------------------------------------------------

class TestHealpixSplit:
    def _records(self, n):
        return [{"coadd": f"/path/{i}.fits", "healpix": i, "n_rows": 100} for i in range(n)]

    def test_disjoint(self):
        recs = self._records(100)
        train, val = split_records_by_healpix(recs, holdout_frac=0.1, seed=0)
        train_coadds = {r["coadd"] for r in train}
        val_coadds = {r["coadd"] for r in val}
        assert train_coadds.isdisjoint(val_coadds)
        assert len(train_coadds) + len(val_coadds) == 100

    def test_sizes(self):
        recs = self._records(100)
        train, val = split_records_by_healpix(recs, holdout_frac=0.05, seed=0)
        assert len(val) == 5
        assert len(train) == 95

    def test_deterministic_with_seed(self):
        recs = self._records(50)
        t1, v1 = split_records_by_healpix(recs, holdout_frac=0.1, seed=42)
        t2, v2 = split_records_by_healpix(recs, holdout_frac=0.1, seed=42)
        assert [r["healpix"] for r in t1] == [r["healpix"] for r in t2]
        assert [r["healpix"] for r in v1] == [r["healpix"] for r in v2]

    def test_invalid_fraction_raises(self):
        recs = self._records(10)
        with pytest.raises(ValueError):
            split_records_by_healpix(recs, holdout_frac=0.0, seed=0)
        with pytest.raises(ValueError):
            split_records_by_healpix(recs, holdout_frac=1.0, seed=0)


# ---------------------------------------------------------------------------
# Redshift-weight behavior — tests 13-14 (model-level)
# ---------------------------------------------------------------------------

class TestRedshiftWeight:
    def _model(self):
        return SpectrumTransformer(
            vocab_size=TOTAL_VOCAB_SIZE,
            d_model=64, n_encoder_layers=1, n_decoder_layers=1,
            n_heads=4, max_seq_len=64, dropout=0.0,
        )

    def test_weight_changes_loss(self):
        torch.manual_seed(0)
        model = self._model()
        enc = torch.randint(0, TOTAL_VOCAB_SIZE, (2, 10))
        dec = torch.randint(0, TOTAL_VOCAB_SIZE, (2, 8))
        tgt = torch.randint(0, TOTAL_VOCAB_SIZE, (2, 8))
        _, loss_w1 = model(enc, dec, targets=tgt, redshift_weight=1.0)
        _, loss_w50 = model(enc, dec, targets=tgt, redshift_weight=50.0)
        # difference should be (50-1) * loss_red_mean = 49 * something positive
        assert loss_w50.item() > loss_w1.item()
        # And specifically a positive difference
        assert (loss_w50 - loss_w1).item() > 0.0

    def test_no_targets_ignores_kwarg(self):
        model = self._model()
        enc = torch.randint(0, TOTAL_VOCAB_SIZE, (1, 6))
        dec = torch.randint(0, TOTAL_VOCAB_SIZE, (1, 4))
        logits, loss = model(enc, dec, redshift_weight=999.0)
        assert loss is None
        assert logits.shape == (1, 4, TOTAL_VOCAB_SIZE)


# ---------------------------------------------------------------------------
# AR eval glue — test 15
# ---------------------------------------------------------------------------

class TestEvaluateAR:
    def test_returns_metrics(self):
        # A tiny model + tiny FakeSpecTok works in CPU and exercises the path.
        spec, z = FakeSpecTok(n_tokens=4, codebook=8), FakeZTok()
        model = SpectrumTransformer(
            vocab_size=TOTAL_VOCAB_SIZE,
            d_model=32, n_encoder_layers=1, n_decoder_layers=1,
            n_heads=4, max_seq_len=64, dropout=0.0,
        ).eval()

        # Build a single batch via a one-shot iterable.
        raw = _make_raw_batch(B=2)
        loader = [raw]

        out = evaluate_ar(
            model, loader, spec, z, "a", torch.device("cpu"),
            max_batches=1, encoder_mask_ratio=0.0,
        )
        # Returns the expected fields; exact values are model-random, not asserted.
        assert "ar_redshift_acc" in out
        assert "ar_spectrum_acc" in out
        assert out["n_samples"] == 2


# ---------------------------------------------------------------------------
# Wandb env-var forcing — tests 16-17
# ---------------------------------------------------------------------------

class TestWandbInit:
    def test_disabled_returns_none(self, tmp_path):
        result = init_wandb(
            mode="disabled",
            project="dummy",
            run_name="r",
            config={},
            out_dir=tmp_path,
        )
        assert result is None

    def test_mode_env_forced_when_online(self, tmp_path, monkeypatch):
        # Pre-set WANDB_MODE to "offline" (mimicking NERSC) and verify
        # init_wandb sets it back to "online" before init.
        monkeypatch.setenv("WANDB_MODE", "offline")
        monkeypatch.setenv("WANDB_API_KEY", "dummy-key")

        captured = {}

        def fake_init(**kw):
            captured["mode"] = kw.get("mode")
            captured["env_mode"] = os.environ.get("WANDB_MODE")
            class _R: url = None
            return _R()

        # Patch wandb import inside the function. The function does
        # `import wandb` at call time, so patch via sys.modules.
        import types
        wb = types.ModuleType("wandb")
        wb.init = fake_init
        monkeypatch.setitem(sys.modules, "wandb", wb)

        init_wandb(
            mode="online",
            project="dummy",
            run_name="r",
            config={},
            out_dir=tmp_path,
        )
        assert os.environ["WANDB_MODE"] == "online"
        assert captured["env_mode"] == "online"
        assert captured["mode"] == "online"

    def test_missing_api_key_falls_back_to_offline(self, tmp_path, monkeypatch):
        monkeypatch.delenv("WANDB_API_KEY", raising=False)

        captured = {}

        def fake_init(**kw):
            captured["mode"] = kw.get("mode")
            class _R: url = None
            return _R()

        import types
        wb = types.ModuleType("wandb")
        wb.init = fake_init
        monkeypatch.setitem(sys.modules, "wandb", wb)

        init_wandb(
            mode="online",
            project="dummy",
            run_name="r",
            config={},
            out_dir=tmp_path,
        )
        assert captured["mode"] == "offline"
        assert os.environ.get("WANDB_MODE") == "offline"


# ---------------------------------------------------------------------------
# log_model_artifact: keep-only-latest prune behaviour
# ---------------------------------------------------------------------------

class _FakeArt:
    """Minimal stand-in for wandb.Artifact / ArtifactVersion."""
    def __init__(self, version: str, aliases=None):
        self.version = version
        self.aliases = list(aliases or [])
        self.name = f"art:{version}"
        self.deleted = False
        self.alias_saves = 0

    def add_file(self, path):
        self.path = path

    def wait(self):
        return self

    def save(self):
        self.alias_saves += 1

    def delete(self):
        self.deleted = True


class _FakeRun:
    def __init__(self, entity="ent", project="proj"):
        self.entity = entity
        self.project = project
        self.logged = []

    def log_artifact(self, art, aliases=None):
        self.logged.append((art, list(aliases or [])))


class TestLogModelArtifactPrune:
    def _setup_wandb_module(self, monkeypatch, existing_versions, new_version):
        """Install a fake `wandb` module that returns existing_versions
        on api.artifact_versions, and constructs new artifacts with new_version."""
        import types

        new_art = _FakeArt(version=new_version)

        def Artifact(name, type, metadata=None):
            new_art.config_name = name
            new_art.type = type
            new_art.metadata = metadata
            return new_art

        class _Api:
            def artifact_versions(self, type_, full_name):
                return list(existing_versions)

        wb = types.ModuleType("wandb")
        wb.Artifact = Artifact
        wb.Api = _Api
        monkeypatch.setitem(sys.modules, "wandb", wb)
        return new_art

    def test_run_none_returns_none(self, tmp_path):
        ckpt = tmp_path / "best.pt"
        ckpt.write_bytes(b"x")
        assert log_model_artifact(None, ckpt, "n") is None

    def test_prune_deletes_old_versions(self, tmp_path, monkeypatch):
        ckpt = tmp_path / "best.pt"
        ckpt.write_bytes(b"x")
        old = [_FakeArt("v0"), _FakeArt("v1", aliases=["latest"])]
        new_art = self._setup_wandb_module(
            monkeypatch, existing_versions=old + [_FakeArt("v2")], new_version="v2"
        )
        run = _FakeRun()
        result = log_model_artifact(
            run, ckpt, "approach_a", aliases=["best"], keep_only_latest=True
        )
        assert result is new_art
        # Old v0 and v1 deleted; v2 (current) untouched.
        assert old[0].deleted is True
        assert old[1].deleted is True
        # The aliased "latest" version had its alias stripped before delete.
        assert old[1].aliases == []
        assert old[1].alias_saves == 1

    def test_keep_only_latest_false_does_not_prune(self, tmp_path, monkeypatch):
        ckpt = tmp_path / "best.pt"
        ckpt.write_bytes(b"x")
        old = [_FakeArt("v0"), _FakeArt("v1")]
        self._setup_wandb_module(
            monkeypatch, existing_versions=old + [_FakeArt("v2")], new_version="v2"
        )
        run = _FakeRun()
        log_model_artifact(
            run, ckpt, "approach_a", aliases=["best"], keep_only_latest=False
        )
        assert old[0].deleted is False
        assert old[1].deleted is False

    def test_prune_first_upload_no_op(self, tmp_path, monkeypatch):
        ckpt = tmp_path / "best.pt"
        ckpt.write_bytes(b"x")
        new_art = self._setup_wandb_module(
            monkeypatch, existing_versions=[_FakeArt("v0")], new_version="v0"
        )
        run = _FakeRun()
        result = log_model_artifact(run, ckpt, "approach_a", keep_only_latest=True)
        assert result is new_art
        # The only existing version equals keep_version → nothing deleted.


# ---------------------------------------------------------------------------
# Redshift masking — tests for stochastic rz masking in Approach A
# ---------------------------------------------------------------------------

class TestRedshiftMasking:
    """Tests for stochastic redshift masking in Approach A."""

    def test_zero_ratio_no_rz_mask(self):
        """encoder_mask_ratio=0.0 should return rz_mask=None."""
        spec, z = FakeSpecTok(), FakeZTok()
        raw = _make_raw_batch()
        enc, dec, tgt, mp, rz_mask = tokenize_and_build(
            raw, spec, z, "a", torch.device("cpu"), encoder_mask_ratio=0.0
        )
        assert rz_mask is None

    def test_full_ratio_all_rz_masked(self):
        """encoder_mask_ratio=1.0 should mask ALL redshift tokens."""
        spec, z = FakeSpecTok(n_tokens=8), FakeZTok()
        raw = _make_raw_batch(B=4)
        enc, dec, tgt, mp, rz_mask = tokenize_and_build(
            raw, spec, z, "a", torch.device("cpu"), encoder_mask_ratio=1.0
        )
        assert rz_mask is not None
        assert rz_mask.shape == (4, 1)
        assert rz_mask.all()
        # Encoder position 1 (redshift) should be MASK_TOKEN
        assert (enc[:, 1] == MASK_TOKEN).all()
        # SOS and EOS should NOT be MASK
        assert (enc[:, 0] == SOS_TOKEN).all()
        assert (enc[:, -1] == EOS_TOKEN).all()

    def test_rz_masking_does_not_affect_decoder_or_target(self):
        """Decoder input and target should be identical regardless of masking."""
        spec, z = FakeSpecTok(), FakeZTok()
        raw = _make_raw_batch()
        torch.manual_seed(0)
        _, dec_a, tgt_a, _, _ = tokenize_and_build(
            raw, spec, z, "a", torch.device("cpu"), encoder_mask_ratio=0.5
        )
        torch.manual_seed(0)
        _, dec_b, tgt_b, _, _ = tokenize_and_build(
            raw, spec, z, "a", torch.device("cpu"), encoder_mask_ratio=0.0
        )
        assert torch.equal(dec_a, dec_b)
        assert torch.equal(tgt_a, tgt_b)

    def test_approach_b_returns_no_rz_mask(self):
        """Approach B should never return rz_mask (no rz in encoder)."""
        spec, z = FakeSpecTok(), FakeZTok()
        raw = _make_raw_batch()
        enc, dec, tgt, mp, rz_mask = tokenize_and_build(
            raw, spec, z, "b", torch.device("cpu"), encoder_mask_ratio=0.5
        )
        assert rz_mask is None

    def test_stochastic_masking_partial(self):
        """With ratio=0.5, some samples should be masked, some not."""
        spec, z = FakeSpecTok(), FakeZTok()
        raw = _make_raw_batch(B=100)
        rng = torch.Generator().manual_seed(42)
        enc, dec, tgt, mp, rz_mask = tokenize_and_build(
            raw, spec, z, "a", torch.device("cpu"), encoder_mask_ratio=0.5,
            rng=rng,
        )
        assert rz_mask is not None
        n_masked = int(rz_mask.sum().item())
        # Should be roughly 50 out of 100 (allow wide margin)
        assert 20 < n_masked < 80

    def test_encoder_layout_approach_a_with_masking(self):
        """Encoder layout should be [SOS, rz_or_mask, s1..sN, EOS]."""
        spec, z = FakeSpecTok(n_tokens=8), FakeZTok()
        raw = _make_raw_batch(B=2)
        enc, _, _, _, rz_mask = tokenize_and_build(
            raw, spec, z, "a", torch.device("cpu"), encoder_mask_ratio=1.0
        )
        assert enc.shape == (2, 1 + 1 + 8 + 1)  # SOS + rz + spec + EOS
        assert (enc[:, 0] == SOS_TOKEN).all()
        assert (enc[:, 1] == MASK_TOKEN).all()  # rz masked
        assert (enc[:, 2:10] == MASK_TOKEN).all()  # spectrum also masked at 1.0
        assert (enc[:, -1] == EOS_TOKEN).all()


# ---------------------------------------------------------------------------
# compute_masked_redshift_acc — tests
# ---------------------------------------------------------------------------

class TestMaskedRedshiftAcc:
    """Tests for compute_masked_redshift_acc."""

    def _craft(self, B=2, V=64):
        logits = torch.zeros(B, 10, V)
        target = torch.zeros(B, 10, dtype=torch.long)
        target[:, 0] = 50  # redshift token at position 0
        target[:, 1:9] = torch.arange(8).unsqueeze(0).expand(B, -1) + 10
        target[:, -1] = EOS_TOKEN
        rz_mask = torch.zeros(B, 1, dtype=torch.bool)
        return logits, target, rz_mask

    def test_all_correct_when_logits_match(self):
        B = 2
        logits, target, rz_mask = self._craft()
        for b in range(B):
            logits[b, 0, target[b, 0]] = 1.0
        rz_mask.fill_(True)
        out = compute_masked_redshift_acc(logits, target, rz_mask)
        assert out["n_rz_masked"] > 0
        assert out["redshift_acc_masked"] == pytest.approx(1.0)

    def test_all_wrong(self):
        logits, target, rz_mask = self._craft(V=64)
        rz_mask.fill_(True)
        out = compute_masked_redshift_acc(logits, target, rz_mask)
        assert out["redshift_acc_masked"] == pytest.approx(0.0)

    def test_zero_mask_returns_nan(self):
        logits, target, rz_mask = self._craft()
        out = compute_masked_redshift_acc(logits, target, rz_mask)
        assert out["n_rz_masked"] == 0
        assert out["redshift_acc_masked"] != out["redshift_acc_masked"]  # NaN

    def test_partial_mask_only_counts_masked(self):
        logits, target, rz_mask = self._craft(B=4)
        # Make logits correct for all
        for b in range(4):
            logits[b, 0, target[b, 0]] = 1.0
        # Only mask half
        rz_mask[0, 0] = True
        rz_mask[1, 0] = True
        out = compute_masked_redshift_acc(logits, target, rz_mask)
        assert out["n_rz_masked"] == 2
        assert out["redshift_acc_masked"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# AION benchmark metrics — AUC and R² (tests 18+)
# ---------------------------------------------------------------------------

class TestAUCMetrics:
    """Tests for compute_masked_auc and compute_all_auc.

    These tests validate behavior when sklearn IS available. They are
    skipped (pass) when sklearn is absent — the code already gracefully
    returns NaN in that case, so the important invariants (correct dict
    keys, correct n=0 handling) are covered by the no_masked_returns_nan
    and no_positions_returns_nan tests.
    """

    @staticmethod
    def _sklearn_available():
        try:
            from sklearn.metrics import roc_auc_score
            return True
        except ImportError:
            return False

    def _craft(self, B=2, T=8, V=64):
        logits = torch.zeros(B, T + 2, V)
        target = torch.zeros(B, T + 2, dtype=torch.long)
        target[:, 0] = 50
        target[:, 1:1 + T] = torch.arange(T).unsqueeze(0).expand(B, -1) + 10
        target[:, -1] = EOS_TOKEN
        masked = torch.zeros(B, T, dtype=torch.bool)
        return logits, target, masked

    def test_masked_auc_all_correct(self):
        if not self._sklearn_available():
            pytest.skip("sklearn not available")
        logits, target, mp = self._craft()
        for b in range(target.shape[0]):
            for t in range(target.shape[1]):
                logits[b, t, target[b, t]] = 10.0
        mp.fill_(True)
        out = compute_masked_auc(logits, target, mp)
        assert out["n_masked"] > 0
        assert out["mean_mask_auc"] == pytest.approx(1.0, abs=0.05)

    def test_masked_auc_all_wrong(self):
        if not self._sklearn_available():
            pytest.skip("sklearn not available")
        logits, target, mp = self._craft(V=64)
        mp.fill_(True)
        for b in range(target.shape[0]):
            for t in range(target.shape[1]):
                correct_tok = target[b, t].item()
                logits[b, t, correct_tok] = -100.0
        out = compute_masked_auc(logits, target, mp)
        assert out["n_masked"] > 0
        assert out["mean_mask_auc"] == pytest.approx(0.0, abs=0.05)

    def test_masked_auc_no_masked_returns_nan(self):
        logits, target, mp = self._craft()
        out = compute_masked_auc(logits, target, mp)
        assert out["n_masked"] == 0
        assert out["mean_mask_auc"] != out["mean_mask_auc"]

    def test_all_auc_all_correct(self):
        if not self._sklearn_available():
            pytest.skip("sklearn not available")
        logits, target, mp = self._craft()
        for b in range(target.shape[0]):
            for t in range(target.shape[1]):
                logits[b, t, target[b, t]] = 10.0
        out = compute_all_auc(logits, target)
        assert out["n_positions"] > 0
        assert out["all_mean_auc"] == pytest.approx(1.0, abs=0.05)

    def test_all_auc_no_positions_returns_nan(self):
        logits = torch.zeros(2, 10, 64)
        target = torch.full((2, 10), -100, dtype=torch.long)
        out = compute_all_auc(logits, target)
        assert out["n_positions"] == 0
        assert out["all_mean_auc"] != out["all_mean_auc"]


class TestR2Metrics:
    """Tests for compute_masked_r2 and compute_all_r2."""

    def _craft(self, B=2, T=8, V=64):
        logits = torch.zeros(B, T + 2, V)
        target = torch.zeros(B, T + 2, dtype=torch.long)
        target[:, 0] = 50
        target[:, 1:1 + T] = torch.arange(T).unsqueeze(0).expand(B, -1) + 10
        target[:, -1] = EOS_TOKEN
        masked = torch.zeros(B, T, dtype=torch.bool)
        return logits, target, masked

    def test_masked_r2_perfect(self):
        logits, target, mp = self._craft()
        for b in range(target.shape[0]):
            for t in range(target.shape[1]):
                logits[b, t, target[b, t]] = 10.0
        mp.fill_(True)
        out = compute_masked_r2(logits, target, mp)
        assert out["n_masked"] > 0
        assert out["masked_spec_r2"] == pytest.approx(1.0, abs=0.01)

    def test_masked_r2_random_logits(self):
        torch.manual_seed(42)
        logits, target, mp = self._craft()
        logits.normal_(std=1.0)
        mp.fill_(True)
        out = compute_masked_r2(logits, target, mp)
        assert out["n_masked"] > 0
        # Random logits should give R² near 0 (within noise)
        assert -0.2 < out["masked_spec_r2"] < 0.3

    def test_masked_r2_no_masked_returns_nan(self):
        logits, target, mp = self._craft()
        out = compute_masked_r2(logits, target, mp)
        assert out["n_masked"] == 0
        assert out["masked_spec_r2"] != out["masked_spec_r2"]  # NaN

    def test_all_r2_perfect(self):
        logits, target, mp = self._craft()
        for b in range(target.shape[0]):
            for t in range(target.shape[1]):
                logits[b, t, target[b, t]] = 10.0
        out = compute_all_r2(logits, target)
        assert out["n_positions"] > 0
        assert out["all_spec_r2"] == pytest.approx(1.0, abs=0.01)

    def test_all_r2_no_positions_returns_nan(self):
        logits = torch.zeros(2, 10, 64)
        target = torch.full((2, 10), -100, dtype=torch.long)
        out = compute_all_r2(logits, target)
        assert out["n_positions"] == 0
        assert out["all_spec_r2"] != out["all_spec_r2"]  # NaN


class TestEvaluateARReturnsMetrics:
    """Test that evaluate_ar returns the new AUC/R² metrics."""

    def test_returns_auc_and_r2_metrics(self):
        spec, z = FakeSpecTok(n_tokens=4, codebook=8), FakeZTok()
        model = SpectrumTransformer(
            vocab_size=TOTAL_VOCAB_SIZE,
            d_model=32, n_encoder_layers=1, n_decoder_layers=1,
            n_heads=4, max_seq_len=64, dropout=0.0,
        ).eval()

        raw = _make_raw_batch(B=2)
        loader = [raw]

        out = evaluate_ar(
            model, loader, spec, z, "a", torch.device("cpu"),
            max_batches=1, encoder_mask_ratio=0.0,
        )
        # AR metrics (masked ones are NaN since no masking in AR)
        assert "ar_mean_mask_auc" in out
        assert "ar_masked_spec_r2" in out
        # AR all-position metrics are always computed
        assert "ar_all_mean_auc" in out
        assert "ar_all_spec_r2" in out
        # Original metrics still present
        assert "ar_redshift_acc" in out
        assert "ar_spectrum_acc" in out
        assert out["n_samples"] == 2
