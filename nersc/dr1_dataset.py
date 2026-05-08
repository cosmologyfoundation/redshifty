"""
DR1IndexedDataset
=================
Streaming PyTorch Dataset over a DR1 manifest produced by build_dr1_index.py.

Each manifest line is one healpix coadd. Each coadd holds N spectra. We
expand the manifest to one (coadd_path, row_idx) pair per spectrum and read
spectra on demand -- so the working set stays bounded even with all of DR1.

For tokenizer pretraining we don't need redshift or the full collation
machinery from src.utils.data; we only need (flux, ivar) on the native
stitched grid. We reuse `stitch_bands` from the existing module.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from astropy.io import fits
from torch.utils.data import Dataset

# The repo's existing helpers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.data import stitch_bands  # noqa: E402


def load_manifest(path: Path) -> List[dict]:
    """Read a JSONL manifest into a list of dicts."""
    out = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


class DR1IndexedDataset(Dataset):
    """One spectrum per index, opened on demand from DR1 healpix coadds.

    Args:
        manifest: list of records (coadd, redrock, n_rows, ...) from
            build_dr1_index.py.
        require_good_zwarn: drop rows with ZWARN != 0 or bad fiber status.
        require_nonzero_flux: drop all-zero spectra.
        max_spectra: cap on total spectra (after filtering); used for smoke.
        cache_size: how many recently-touched FITS HDULs to keep open in
            this worker. With shuffled DataLoader this should be 1; with
            sequential reads, larger helps.
    """

    def __init__(
        self,
        manifest: List[dict],
        require_good_zwarn: bool = True,
        require_nonzero_flux: bool = True,
        max_spectra: Optional[int] = None,
        cache_size: int = 1,
    ):
        self.records = manifest
        self.require_good_zwarn = require_good_zwarn
        self.require_nonzero_flux = require_nonzero_flux
        self.cache_size = max(1, cache_size)

        self.flat_index: List[Tuple[int, int]] = []
        for rec_idx, rec in enumerate(self.records):
            n = rec.get("n_rows", -1)
            if n <= 0:
                # We didn't precompute counts -- read header now.
                with fits.open(rec["coadd"], memmap=False) as h:
                    n = int(h["FIBERMAP"].header["NAXIS2"])
                rec["n_rows"] = n
            for row in range(n):
                self.flat_index.append((rec_idx, row))
                if max_spectra is not None and len(self.flat_index) >= max_spectra:
                    break
            if max_spectra is not None and len(self.flat_index) >= max_spectra:
                break

        self._hdul_cache: dict = {}
        self._hdul_order: List[int] = []

    def __len__(self) -> int:
        return len(self.flat_index)

    def _open(self, rec_idx: int):
        if rec_idx in self._hdul_cache:
            return self._hdul_cache[rec_idx]
        if len(self._hdul_order) >= self.cache_size:
            evict = self._hdul_order.pop(0)
            try:
                self._hdul_cache.pop(evict).close()
            except Exception:
                pass
        rec = self.records[rec_idx]
        coadd = fits.open(rec["coadd"], memmap=True)
        redrock = fits.open(rec["redrock"], memmap=True)
        self._hdul_cache[rec_idx] = (coadd, redrock)
        self._hdul_order.append(rec_idx)
        return coadd, redrock

    def __getitem__(self, idx):
        rec_idx, row = self.flat_index[idx]
        coadd, redrock = self._open(rec_idx)

        if self.require_good_zwarn:
            zwarn = int(redrock["REDSHIFTS"].data["ZWARN"][row])
            fiberstatus = int(coadd["FIBERMAP"].data["COADD_FIBERSTATUS"][row])
            if zwarn != 0 or fiberstatus != 0:
                # Caller is expected to wrap in a quality-filter Subset; or
                # we return None and the collate skips it. We return None
                # and rely on a custom collate to drop it.
                return None

        b_flux = coadd["B_FLUX"].data[row]
        r_flux = coadd["R_FLUX"].data[row]
        z_flux = coadd["Z_FLUX"].data[row]
        if self.require_nonzero_flux:
            tot = float(np.abs(b_flux).sum() + np.abs(r_flux).sum() + np.abs(z_flux).sum())
            if tot == 0.0:
                return None

        b_ivar = coadd["B_IVAR"].data[row]
        r_ivar = coadd["R_IVAR"].data[row]
        z_ivar = coadd["Z_IVAR"].data[row]

        b_mask = coadd["B_MASK"].data[row] != 0
        r_mask = coadd["R_MASK"].data[row] != 0
        z_mask = coadd["Z_MASK"].data[row] != 0

        b_wave = coadd["B_WAVELENGTH"].data
        r_wave = coadd["R_WAVELENGTH"].data
        z_wave = coadd["Z_WAVELENGTH"].data

        stitched = stitch_bands(
            [b_wave, r_wave, z_wave],
            [b_flux, r_flux, z_flux],
            [b_ivar, r_ivar, z_ivar],
            [b_mask, r_mask, z_mask],
        )

        z = float(redrock["REDSHIFTS"].data["Z"][row])

        return {
            "flux": torch.from_numpy(stitched["flux"].copy()),
            "ivar": torch.from_numpy(stitched["ivar"].copy()),
            "mask": torch.from_numpy(stitched["mask"].copy()),
            "wavelength": torch.from_numpy(stitched["wavelength"].copy()),
            "z": torch.tensor(z, dtype=torch.float32),
        }


def collate_dr1_skip_none(batch):
    """Collate that drops None entries (filtered-out rows) and stacks the rest.

    Spectra in DR1 may have slightly different stitched lengths if mask
    coverage differs. We pad to the longest in the batch with zeros and
    set ivar to zero for padded positions so the tokenizer normalization
    treats them as missing.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    L = max(b["flux"].shape[0] for b in batch)

    def _pad(t, fill=0.0):
        if t.shape[0] == L:
            return t
        out = torch.full((L,), fill, dtype=t.dtype)
        out[: t.shape[0]] = t
        return out

    return {
        "flux": torch.stack([_pad(b["flux"]) for b in batch]),
        "ivar": torch.stack([_pad(b["ivar"], 0.0) for b in batch]),
        "mask": torch.stack([_pad(b["mask"].to(torch.bool), True) for b in batch]),
        "z": torch.stack([b["z"] for b in batch]),
    }
