"""
Stage a DR1 manifest's healpix files from $CFS to $SCRATCH.

Reads a manifest JSONL (from build_dr1_index.py), copies each
(coadd, redrock) pair to a SCRATCH mirror that preserves the source path
structure, and writes a new manifest pointing at the SCRATCH paths.

Why: $CFS is high-capacity but not tuned for random small-file reads in
parallel jobs. $SCRATCH is Lustre and an order of magnitude faster for
the FITS-open-per-spectrum I/O pattern. Staging once before a long run
removes the I/O bottleneck (see RESEARCH_LOG.md Phase 8 — pretrain was
~80% I/O-bound).

Usage:
    python nersc/stage_to_scratch.py \\
        --src-manifest $SCRATCH/deepsrch/manifests/dr1_200.jsonl \\
        --dst-root     $SCRATCH/deepsrch/dr1_staged \\
        --dst-manifest $SCRATCH/deepsrch/manifests/dr1_200_scratch.jsonl

Then train as usual but point at the new manifest:
    MANIFEST=$SCRATCH/deepsrch/manifests/dr1_200_scratch.jsonl \\
        sbatch nersc/train_transformer.slurm

Idempotent: re-running skips files already present at dst with matching size.

Run from a compute node via stage_to_scratch.slurm OR from a login node
for small (~50 GB) manifests.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Stage DR1 healpix files CFS->SCRATCH")
    p.add_argument("--src-manifest", type=Path, required=True,
                   help="JSONL manifest from build_dr1_index.py (CFS paths)")
    p.add_argument("--dst-root", type=Path, required=True,
                   help="Root directory on $SCRATCH where files are mirrored")
    p.add_argument("--dst-manifest", type=Path, required=True,
                   help="Output manifest with SCRATCH paths")
    p.add_argument("--max-files", type=int, default=None,
                   help="Cap on number of healpix dirs to stage (for testing)")
    p.add_argument("--src-prefix", type=str,
                   default="/global/cfs/cdirs/desi/public/dr1/",
                   help="Source path prefix to strip when computing destination layout")
    return p.parse_args()


def stage_one(src: Path, dst: Path) -> bool:
    """Copy src -> dst preserving size/mtime. Skip if dst exists with matching size."""
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return False  # skipped
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main():
    args = parse_args()

    if not args.src_manifest.is_file():
        print(f"ERROR: missing src manifest {args.src_manifest}", file=sys.stderr)
        sys.exit(1)

    src_prefix = args.src_prefix
    dst_root = args.dst_root
    dst_root.mkdir(parents=True, exist_ok=True)
    args.dst_manifest.parent.mkdir(parents=True, exist_ok=True)

    out_records = []
    n_copied = 0
    n_skipped = 0
    n_bytes = 0
    t0 = time.time()

    with args.src_manifest.open() as f:
        records = [json.loads(line) for line in f if line.strip()]
    if args.max_files is not None:
        records = records[: args.max_files]

    print(f"[stage] {len(records)} records  src_prefix={src_prefix}  dst_root={dst_root}")

    for k, rec in enumerate(records):
        new_rec = dict(rec)
        for key in ("coadd", "redrock"):
            src = Path(rec[key])
            try:
                rel = src.relative_to(src_prefix)
            except ValueError:
                # Path didn't share the prefix; preserve full structure under dst_root
                rel = Path(str(src).lstrip("/"))
            dst = dst_root / rel
            try:
                copied = stage_one(src, dst)
            except Exception as e:
                print(f"  ERROR copying {src} -> {dst}: {e}", file=sys.stderr)
                continue
            if copied:
                n_copied += 1
                n_bytes += dst.stat().st_size
            else:
                n_skipped += 1
            new_rec[key] = str(dst)
        out_records.append(new_rec)

        if (k + 1) % 50 == 0:
            dt = time.time() - t0
            mb = n_bytes / 1e6
            print(f"  {k+1}/{len(records)}  copied={n_copied} "
                  f"skipped={n_skipped}  {mb:.1f} MB  {mb/max(dt,1):.1f} MB/s")

    with args.dst_manifest.open("w") as f:
        for rec in out_records:
            f.write(json.dumps(rec) + "\n")

    dt = time.time() - t0
    mb = n_bytes / 1e6
    print(f"\n[stage] done. {len(out_records)} records -> {args.dst_manifest}")
    print(f"[stage] copied {n_copied} files ({mb:.1f} MB) in {dt:.1f}s "
          f"({mb/max(dt,1):.1f} MB/s); skipped {n_skipped} (already present)")


if __name__ == "__main__":
    main()
