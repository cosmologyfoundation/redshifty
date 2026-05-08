"""
DR1TokenizedDataset
===================
Wraps DR1IndexedDataset and produces transformer-ready sequences:
(encoder_input, decoder_input, target) per Approach A or B.

Mirrors src.datasets.tokenized_dataset.TokenizedSpectrumDataset but
reads spectra on demand from the manifest instead of holding everything
in memory.

The spectrum tokenizer is run in eval mode with no grad. The redshift
tokenizer must be `fit()` on a representative redshift sample before
this dataset is used.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from dr1_dataset import DR1IndexedDataset  # noqa: E402
from src.tokenizers.spectrum import SpectrumTokenizer  # noqa: E402
from src.tokenizers.redshift import RedshiftTokenizer  # noqa: E402
from src.models.transformer import (  # noqa: E402
    PAD_TOKEN,
    build_approach_a_sequences,
    build_approach_b_sequences,
    encode_redshift_token,
    encode_spectrum_token,
)


class DR1TokenizedDataset(Dataset):
    """Per-item tokenization of a DR1IndexedDataset.

    Args:
        base: an underlying DR1IndexedDataset.
        spectrum_tokenizer: a SpectrumTokenizer in eval mode (caller's
            responsibility to load weights and set requires_grad=False).
        redshift_tokenizer: a fitted RedshiftTokenizer.
        approach: 'a' (joint) or 'b' (masked redshift).
        device: device on which to run the tokenizer encode pass.
            Tokenized outputs are returned on CPU for the DataLoader
            to pin/transfer.
    """

    def __init__(
        self,
        base: DR1IndexedDataset,
        spectrum_tokenizer: SpectrumTokenizer,
        redshift_tokenizer: RedshiftTokenizer,
        approach: str = "a",
        device: torch.device = torch.device("cpu"),
    ):
        approach = approach.lower()
        assert approach in ("a", "b"), f"approach must be 'a' or 'b', got {approach!r}"
        self.base = base
        self.spectrum_tokenizer = spectrum_tokenizer
        self.redshift_tokenizer = redshift_tokenizer
        self.approach = approach
        self.device = device

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        spec = self.base[idx]
        if spec is None:
            return None

        flux = spec["flux"].unsqueeze(0).to(self.device, non_blocking=True)
        ivar = spec["ivar"].unsqueeze(0).to(self.device, non_blocking=True)
        istd = torch.sqrt(ivar.clamp(min=1e-10))
        x = torch.stack([flux, istd], dim=1)

        with torch.no_grad():
            indices, _ = self.spectrum_tokenizer.encode(x)

        spectrum_tokens = encode_spectrum_token(indices.squeeze(0)).cpu()

        z = float(spec["z"])
        redshift_idx = self.redshift_tokenizer.encode(z)
        redshift_token = encode_redshift_token(redshift_idx).cpu()

        if self.approach == "a":
            enc, dec_in, target = build_approach_a_sequences(redshift_token, spectrum_tokens)
        else:
            enc, dec_in, target = build_approach_b_sequences(redshift_token, spectrum_tokens)

        return {
            "encoder_input": enc,
            "decoder_input": dec_in,
            "target": target,
            "redshift": torch.tensor(z, dtype=torch.float32),
        }


def collate_tokenized_skip_none(batch):
    """Drop None entries (filtered rows), pad sequences, return dict of tensors."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    n = len(batch)
    max_enc = max(len(b["encoder_input"]) for b in batch)
    max_dec = max(len(b["decoder_input"]) for b in batch)
    max_tgt = max(len(b["target"]) for b in batch)

    encoder_input = torch.full((n, max_enc), PAD_TOKEN, dtype=torch.long)
    encoder_mask = torch.zeros(n, max_enc, dtype=torch.bool)  # True = padding
    decoder_input = torch.full((n, max_dec), PAD_TOKEN, dtype=torch.long)
    decoder_mask = torch.zeros(n, max_dec, dtype=torch.bool)
    target = torch.full((n, max_tgt), -100, dtype=torch.long)

    for i, b in enumerate(batch):
        le, ld, lt = len(b["encoder_input"]), len(b["decoder_input"]), len(b["target"])
        encoder_input[i, :le] = b["encoder_input"]
        encoder_mask[i, le:] = True
        decoder_input[i, :ld] = b["decoder_input"]
        decoder_mask[i, ld:] = True
        target[i, :lt] = b["target"]

    return {
        "encoder_input": encoder_input,
        "decoder_input": decoder_input,
        "target": target,
        "encoder_mask": encoder_mask,
        "decoder_mask": decoder_mask,
        "redshift": torch.stack([b["redshift"] for b in batch]),
    }


def collect_redshifts(records, max_files=None):
    """Open redrock files in a manifest and pull all Z values.

    Used to fit RedshiftTokenizer before training. Cheap -- only reads
    the REDSHIFTS table's Z column.

    Args:
        records: list of manifest records (from load_manifest)
        max_files: optionally cap how many files to scan (subsample)

    Returns:
        torch.Tensor of redshifts (1D)
    """
    from astropy.io import fits

    zs = []
    n_scan = len(records) if max_files is None else min(max_files, len(records))
    for rec in records[:n_scan]:
        with fits.open(rec["redrock"], memmap=True) as h:
            z = h["REDSHIFTS"].data["Z"]
            zwarn = h["REDSHIFTS"].data["ZWARN"]
            good = zwarn == 0
            zs.append(z[good].astype("float32").copy())
    if not zs:
        return torch.zeros(0)
    import numpy as np
    return torch.from_numpy(np.concatenate(zs))
