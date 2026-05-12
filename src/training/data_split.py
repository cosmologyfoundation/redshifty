"""
Dataset-agnostic split helpers.

`split_records_by_healpix`: partition a manifest (list of dicts, each
with a `coadd` or `healpix` field) into train/val by *record*, not by
row. This avoids same-pointing leakage where spectra from the same
healpix file land in both train and val.

The function is intentionally type-agnostic about records — it accepts
any list of dicts and partitions by index. Caller decides what's a
"record" (here, one record = one healpix coadd file).
"""

from __future__ import annotations

from typing import List, Tuple

import torch


def split_records_by_healpix(
    records: List[dict],
    holdout_frac: float,
    seed: int = 42,
) -> Tuple[List[dict], List[dict]]:
    """Partition records into (train, val) by record, not by row.

    Args:
        records: list of manifest records (each typically has 'coadd',
            'redrock', 'healpix' fields).
        holdout_frac: fraction of records to put in val. Must be in (0, 1).
        seed: torch RNG seed for the random permutation.

    Returns:
        (train_records, val_records). Disjoint. val is the LAST
        `ceil(N * holdout_frac)` after a seeded shuffle, so changing
        `holdout_frac` re-uses the same shuffled order — adding more
        records is monotonic in val growth.

    Raises:
        ValueError if `holdout_frac` is not in (0, 1) or if the
        resulting val set would be empty.
    """
    n = len(records)
    if not 0.0 < holdout_frac < 1.0:
        raise ValueError(f"holdout_frac must be in (0,1), got {holdout_frac}")
    n_val = max(1, int(round(n * holdout_frac)))
    if n_val >= n:
        raise ValueError(
            f"holdout_frac={holdout_frac} on {n} records leaves no train set"
        )

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    val_idx = set(perm[-n_val:])
    train = [records[i] for i in range(n) if i not in val_idx]
    val = [records[i] for i in range(n) if i in val_idx]
    return train, val
