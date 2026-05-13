"""
Sequence construction for transformer training.

Given a raw batch (flux, ivar, z) — produced by `DR1IndexedDataset` or
any equivalent dataset — tokenize spectra (frozen `SpectrumTokenizer`)
and redshifts (`RedshiftTokenizer`), then build encoder/decoder/target
sequences for Approach A (encoder sees redshift) or B (encoder masked
of redshift).

Also supports BERT-style encoder masking: replace a fraction of the
encoder's spectrum tokens with `MASK_TOKEN` before the model sees them,
without changing the decoder input or target. Returns the boolean
masked-positions tensor so eval/metrics can report accuracy on the
masked positions only (the honest spectrum-reconstruction number).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from src.models.transformer import (
    EOS_TOKEN,
    MASK_TOKEN,
    REDSHIFT_TOKEN_OFFSET,
    SOS_TOKEN,
    SPECTRUM_TOKEN_OFFSET,
)


def tokenize_and_build(
    raw_batch: dict,
    spec_tok,
    z_tok,
    approach: str,
    device: torch.device,
    encoder_mask_ratio: float = 0.0,
    rng: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Convert a raw spectrum batch into transformer-ready sequences.

    Args:
        raw_batch: dict with "flux" (B, L), "ivar" (B, L), "z" (B,) tensors.
            Produced by `collate_dr1_skip_none` or equivalent.
        spec_tok: a frozen `SpectrumTokenizer` (eval mode).
        z_tok: a fitted `RedshiftTokenizer`.
        approach: 'a' (encoder sees redshift) or 'b' (encoder does not).
        device: where the spectrum tokenizer + transformer live.
        encoder_mask_ratio: fraction of encoder spectrum positions to
            replace with `MASK_TOKEN` (BERT-style). Decoder input and
            target are NOT modified. Default 0.0 = no masking.
        rng: optional `torch.Generator` for reproducible masking.

    Returns:
        encoder_input: (B, L_enc) long tensor.
        decoder_input: (B, L_dec) long tensor (teacher-forced).
        target: (B, L_dec) long tensor.
        masked_positions: (B, T_spec) bool tensor — True where the
            encoder's spectrum tokens were replaced by MASK. None if
            `encoder_mask_ratio == 0.0`. T_spec = number of spectrum
            positions per sequence (e.g. 272 from the tokenizer).
        rz_mask: (B, 1) bool tensor — True where the encoder's redshift
            token was replaced by MASK (approach A only). None if
            `encoder_mask_ratio == 0.0` or approach is "b".
    """
    if approach not in ("a", "b"):
        raise ValueError(f"approach must be 'a' or 'b', got {approach!r}")
    if not 0.0 <= encoder_mask_ratio <= 1.0:
        raise ValueError(f"encoder_mask_ratio must be in [0, 1], got {encoder_mask_ratio}")

    flux = raw_batch["flux"].to(device, non_blocking=True)
    ivar = raw_batch["ivar"].to(device, non_blocking=True)
    z_vals = raw_batch["z"]  # may stay on CPU; encode is per-item

    istd = torch.sqrt(ivar.clamp(min=1e-10))
    x = torch.stack([flux, istd], dim=1)  # (B, 2, L)

    with torch.no_grad():
        spec_indices, _ = spec_tok.encode(x)  # (B, n_tokens) or (B, 1, n_tokens)
    if spec_indices.dim() == 3:
        spec_indices = spec_indices.squeeze(1)
    spec_tokens = spec_indices.long() + SPECTRUM_TOKEN_OFFSET  # (B, T_spec)

    redshift_idx = torch.tensor(
        [z_tok.encode(float(z)) for z in z_vals.tolist()],
        dtype=torch.long, device=device,
    )
    redshift_tokens = redshift_idx + REDSHIFT_TOKEN_OFFSET  # (B,)

    B, T_spec = spec_tokens.shape
    sos = torch.full((B, 1), SOS_TOKEN, dtype=torch.long, device=device)
    eos = torch.full((B, 1), EOS_TOKEN, dtype=torch.long, device=device)
    rz = redshift_tokens.unsqueeze(1)  # (B, 1)

    # Apply masking to the encoder's spectrum tokens only.
    masked_positions: Optional[torch.Tensor] = None
    spec_tokens_enc = spec_tokens
    if encoder_mask_ratio > 0.0:
        if rng is None:
            mask = torch.rand(B, T_spec, device=device) < encoder_mask_ratio
        else:
            mask = torch.rand(B, T_spec, device=device, generator=rng) < encoder_mask_ratio
        masked_positions = mask
        spec_tokens_enc = torch.where(
            mask,
            torch.full_like(spec_tokens, MASK_TOKEN),
            spec_tokens,
        )

    # Stochastically mask the redshift token in the encoder (approach A only).
    rz_mask: Optional[torch.Tensor] = None
    rz_enc = rz
    if approach == "a" and encoder_mask_ratio > 0.0:
        if rng is None:
            rz_mask = torch.rand(B, 1, device=device) < encoder_mask_ratio
        else:
            rz_mask = torch.rand(B, 1, device=device, generator=rng) < encoder_mask_ratio
        rz_enc = torch.where(rz_mask, torch.full_like(rz, MASK_TOKEN), rz)

    if approach == "a":
        encoder_input = torch.cat([sos, rz_enc, spec_tokens_enc, eos], dim=1)
    else:  # 'b'
        encoder_input = torch.cat([sos, spec_tokens_enc, eos], dim=1)

    # Decoder input and target use the UNMASKED spec_tokens.
    decoder_input = torch.cat([sos, rz, spec_tokens], dim=1)
    target = torch.cat([rz, spec_tokens, eos], dim=1)

    return encoder_input, decoder_input, target, masked_positions, rz_mask


def lr_at(step: int, base_lr: float, warmup: int, total: int) -> float:
    """Linear warmup -> cosine decay to 1/10 of base."""
    import math
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))
