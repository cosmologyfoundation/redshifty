"""
Evaluation loops for the spectrum transformer.

- `evaluate`: teacher-forced eval. With `encoder_mask_ratio > 0`, also
  reports `masked_spec_acc` (the honest spectrum metric).
- `evaluate_ar`: autoregressive eval. Decoder starts from [SOS] and
  generates token by token; no teacher forcing, no future-token leakage.
  Slow (T+1 forward passes per sample). Intended for end-of-run + best
  checkpoint, not per-val.

Both functions are dataset-agnostic — they take any iterable that yields
raw batches in the shape expected by `tokenize_and_build`.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from src.models.transformer import (
    EOS_TOKEN,
    REDSHIFT_TOKEN_OFFSET,
    SOS_TOKEN,
    SPECTRUM_TOKEN_OFFSET,
)
from src.training.sequences import tokenize_and_build
from src.training.utils import (
    compute_all_auc,
    compute_all_r2,
    compute_loss_breakdown,
    compute_masked_auc,
    compute_masked_metrics,
    compute_masked_redshift_acc,
    compute_masked_r2,
    compute_metrics,
)


@torch.no_grad()
def evaluate(
    model,
    loader,
    spec_tok,
    z_tok,
    approach: str,
    device: torch.device,
    amp: bool,
    redshift_weight: float,
    aux_redshift_weight: float = 1.0,
    encoder_mask_ratio: float = 0.0,
    max_batches: int = 50,
) -> Dict[str, float]:
    """Teacher-forced eval. Returns a dict of averaged metrics over up to
    `max_batches` batches from `loader`.

    Adds `masked_spec_acc` when `encoder_mask_ratio > 0`.
    """
    model.eval()
    losses = 0.0
    metrics_accum = {"overall_acc": 0.0, "redshift_acc": 0.0, "spectrum_acc": 0.0}
    breakdown_accum = {"loss_redshift": 0.0, "loss_spectrum": 0.0, "loss_total": 0.0}
    masked_acc_accum = 0.0
    masked_n_total = 0
    rz_masked_acc_accum = 0.0
    rz_masked_n_total = 0
    # AION benchmark metrics
    masked_auc_accum = 0.0
    masked_r2_accum = 0.0
    masked_n_for_auc_r2 = 0
    all_auc_accum = 0.0
    all_r2_accum = 0.0
    all_n_total = 0
    n = 0
    for i, raw in enumerate(loader):
        if raw is None:
            continue
        if i >= max_batches:
            break
        enc, dec, tgt, mask_pos, rz_mask = tokenize_and_build(
            raw, spec_tok, z_tok, approach, device,
            encoder_mask_ratio=encoder_mask_ratio,
        )
        with torch.amp.autocast("cuda", enabled=amp):
            logits, loss = model(enc, dec, targets=tgt, redshift_weight=redshift_weight,
                                  aux_redshift_weight=aux_redshift_weight)
        losses += float(loss.item())
        m = compute_metrics(logits, tgt)
        for k in metrics_accum:
            metrics_accum[k] += m[k]
        b = compute_loss_breakdown(logits, tgt)
        for k in breakdown_accum:
            breakdown_accum[k] += b[k]
        if mask_pos is not None:
            mm = compute_masked_metrics(logits, tgt, mask_pos)
            if mm["n_masked"] > 0:
                # weighted average so positions with more masks count more
                masked_acc_accum += mm["masked_spec_acc"] * mm["n_masked"]
                masked_n_total += mm["n_masked"]
        if rz_mask is not None:
            rm = compute_masked_redshift_acc(logits, tgt, rz_mask)
            if rm["n_rz_masked"] > 0:
                rz_masked_acc_accum += rm["redshift_acc_masked"] * rm["n_rz_masked"]
                rz_masked_n_total += rm["n_rz_masked"]
        # AION benchmark: AUC and R² at masked and all positions
        if mask_pos is not None:
            ma = compute_masked_auc(logits, tgt, mask_pos)
            mr = compute_masked_r2(logits, tgt, mask_pos)
            if ma["n_masked"] > 0:
                masked_auc_accum += ma["mean_mask_auc"] * ma["n_masked"]
                masked_r2_accum += mr["masked_spec_r2"] * ma["n_masked"]
                masked_n_for_auc_r2 += ma["n_masked"]
        aa = compute_all_auc(logits, tgt)
        ar = compute_all_r2(logits, tgt)
        if aa["n_positions"] > 0:
            all_auc_accum += aa["all_mean_auc"] * aa["n_positions"]
            all_r2_accum += ar["all_spec_r2"] * aa["n_positions"]
            all_n_total += aa["n_positions"]
        n += 1

    if n == 0:
        nan = float("nan")
        out = {"loss": nan, **{k: nan for k in metrics_accum},
               **{k: nan for k in breakdown_accum}}
        if encoder_mask_ratio > 0.0:
            out["masked_spec_acc"] = nan
            out["redshift_acc_masked"] = nan
            out["mean_mask_auc"] = nan
            out["masked_spec_r2"] = nan
        out["all_mean_auc"] = nan
        out["all_spec_r2"] = nan
        return out

    out = {"loss": losses / n, **{k: v / n for k, v in metrics_accum.items()}}
    out.update({k: v / n for k, v in breakdown_accum.items()})
    if encoder_mask_ratio > 0.0:
        out["masked_spec_acc"] = (
            masked_acc_accum / masked_n_total if masked_n_total > 0 else float("nan")
        )
        out["redshift_acc_masked"] = (
            rz_masked_acc_accum / rz_masked_n_total if rz_masked_n_total > 0 else float("nan")
        )
        out["mean_mask_auc"] = (
            masked_auc_accum / masked_n_for_auc_r2 if masked_n_for_auc_r2 > 0 else float("nan")
        )
        out["masked_spec_r2"] = (
            masked_r2_accum / masked_n_for_auc_r2 if masked_n_for_auc_r2 > 0 else float("nan")
        )
    out["all_mean_auc"] = (
        all_auc_accum / all_n_total if all_n_total > 0 else float("nan")
    )
    out["all_spec_r2"] = (
        all_r2_accum / all_n_total if all_n_total > 0 else float("nan")
    )
    return out


@torch.no_grad()
def evaluate_ar(
    model,
    loader,
    spec_tok,
    z_tok,
    approach: str,
    device: torch.device,
    max_batches: int = 4,
    encoder_mask_ratio: float = 0.0,
) -> Dict[str, float]:
    """Autoregressive eval — no teacher forcing.

    For each sample, generate `T+1` tokens starting from SOS and compare
    against the target. Slow; cap with `max_batches`.

    Returns:
        {'ar_redshift_acc', 'ar_spectrum_acc', 'n_samples'}
    """
    model.eval()
    total_red_correct = 0
    total_spec_correct = 0
    total_spec_positions = 0
    total_samples = 0
    # AION benchmark: AUC and R² (no masking in AR, so masked metrics are NaN)
    all_auc_accum = 0.0
    all_r2_accum = 0.0
    all_n_total = 0

    for i, raw in enumerate(loader):
        if raw is None:
            continue
        if i >= max_batches:
            break

        enc, _dec_unused, tgt, _, _ = tokenize_and_build(
            raw, spec_tok, z_tok, approach, device,
            encoder_mask_ratio=encoder_mask_ratio,
        )
        B, L_dec = tgt.shape
        # generate L_dec tokens; SpectrumTransformer.generate returns
        # (B, 1 + max_new_tokens) including the SOS, so we ask for L_dec.
        generated = model.generate(
            enc,
            decoder_start_token=SOS_TOKEN,
            max_new_tokens=L_dec,
            temperature=1.0,
        )
        # generated[:, 0] is SOS; predictions for tgt[:, j] are at
        # generated[:, j+1]. Trim to first L_dec predictions.
        gen_preds = generated[:, 1:1 + L_dec]
        if gen_preds.shape[1] < L_dec:
            # generate() may have early-stopped on EOS; pad with PAD-equivalent
            # but score only over the positions we have. The simpler approach:
            # use only the overlap length.
            L_dec = gen_preds.shape[1]
            tgt = tgt[:, :L_dec]

        valid = tgt != -100  # (B, L_dec) — typically all True
        # Position 0 is the redshift token.
        total_red_correct += int(((gen_preds[:, 0] == tgt[:, 0]) & valid[:, 0]).sum().item())
        # Positions 1..L_dec-1 are spectrum (+ EOS at the very end).
        if L_dec > 1:
            sp_pred = gen_preds[:, 1:]
            sp_tgt = tgt[:, 1:]
            sp_valid = valid[:, 1:]
            total_spec_correct += int(((sp_pred == sp_tgt) & sp_valid).sum().item())
            total_spec_positions += int(sp_valid.sum().item())
        total_samples += B

        # AUC/R²: run forward pass with generated decoder input to get logits.
        # This measures how well the model ranks correct tokens given what it generated.
        with torch.amp.autocast("cuda", enabled=False):
            logits_ar, _ = model(enc, gen_preds)  # (B, L_dec, V)
        aa = compute_all_auc(logits_ar, tgt)
        ar = compute_all_r2(logits_ar, tgt)
        if aa["n_positions"] > 0:
            all_auc_accum += aa["all_mean_auc"] * aa["n_positions"]
            all_r2_accum += ar["all_spec_r2"] * aa["n_positions"]
            all_n_total += aa["n_positions"]

    if total_samples == 0:
        return {
            "ar_redshift_acc": float("nan"),
            "ar_spectrum_acc": float("nan"),
            "ar_mean_mask_auc": float("nan"),
            "ar_masked_spec_r2": float("nan"),
            "ar_all_mean_auc": float("nan"),
            "ar_all_spec_r2": float("nan"),
            "n_samples": 0,
        }
    ar_red_acc = total_red_correct / total_samples
    ar_spec_acc = (total_spec_correct / total_spec_positions
                   if total_spec_positions > 0 else float("nan"))
    return {
        "ar_redshift_acc": ar_red_acc,
        "ar_spectrum_acc": ar_spec_acc,
        "ar_mean_mask_auc": float("nan"),  # AR uses no masking
        "ar_masked_spec_r2": float("nan"),  # AR uses no masking
        "ar_all_mean_auc": (
            all_auc_accum / all_n_total if all_n_total > 0 else float("nan")
        ),
        "ar_all_spec_r2": (
            all_r2_accum / all_n_total if all_n_total > 0 else float("nan")
        ),
        "n_samples": total_samples,
    }
