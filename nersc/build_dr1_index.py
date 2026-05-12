#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a manifest index of DESI DR1 healpix coadds for streaming.

Scans the public DR1 iron tree once and writes a JSONL where each
line is a (coadd_path, redrock_path, n_rows, survey, program, healpix)
record. Downstream training reads this manifest and opens FITS files
on demand instead of globbing the whole tree at every job start.

Run from a Perlmutter login node (or compute) — purely I/O, no GPU needed:

    python nersc/build_dr1_index.py \\
        --root /global/cfs/cdirs/desi/public/dr1 \\
        --surveys sv3 main \\
        --programs bright dark \\
        --out $SCRATCH/desi_dr1_index.jsonl

Subsetting flags exist for smoke tests (--max-healpix, --hpix-stride).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Avoid astropy at index time -- we only need filenames + a row count.
# We open the BINTABLE header to get NAXIS2 (n rows) without reading data.
try:
    from astropy.io import fits
except ImportError:
    fits = None


IRON_BASE = "spectro/redux/iron/healpix"


def parse_args():
    p = argparse.ArgumentParser(description="Build DR1 healpix manifest")
    p.add_argument("--root", type=Path,
                   default=Path("/global/cfs/cdirs/desi/public/dr1"),
                   help="DR1 root on CFS")
    p.add_argument("--production", default="iron",
                   help="Spectroscopic production directory name")
    p.add_argument("--surveys", nargs="+", default=["sv3", "main"],
                   choices=["sv1", "sv2", "sv3", "main"])
    p.add_argument("--programs", nargs="+", default=["bright", "dark"],
                   choices=["bright", "dark", "backup", "other"])
    p.add_argument("--out", type=Path, required=True,
                   help="Output JSONL manifest path")
    p.add_argument("--max-healpix", type=int, default=None,
                   help="Stop after this many healpix dirs (smoke test)")
    p.add_argument("--hpix-stride", type=int, default=1,
                   help="Take every Nth healpix dir (subsample)")
    p.add_argument("--skip-row-count", action="store_true",
                   help="Skip opening FITS to count rows (faster, but n_rows=-1)")
    return p.parse_args()


def count_rows(coadd: Path) -> int:
    """Read NAXIS2 from the FIBERMAP HDU header without loading data."""
    if fits is None:
        return -1
    try:
        with fits.open(coadd, memmap=False) as h:
            return int(h["FIBERMAP"].header["NAXIS2"])
    except Exception as e:
        print(f"  WARN: could not read row count for {coadd.name}: {e}",
              file=sys.stderr)
        return -1


def walk_program(root: Path, production: str, survey: str, program: str):
    """Yield coadd paths under iron/healpix/{survey}/{program}/."""
    base = root / IRON_BASE.replace("iron", production) / survey / program
    if not base.is_dir():
        print(f"  SKIP missing dir: {base}", file=sys.stderr)
        return
    # Layout: {hpix_group}/{healpix}/coadd-{survey}-{program}-{healpix}.fits
    for grp in sorted(p for p in base.iterdir() if p.is_dir()):
        for hpx in sorted(p for p in grp.iterdir() if p.is_dir()):
            coadd = hpx / f"coadd-{survey}-{program}-{hpx.name}.fits"
            redrock = hpx / f"redrock-{survey}-{program}-{hpx.name}.fits"
            if coadd.is_file() and redrock.is_file():
                yield coadd, redrock, int(hpx.name)


def main():
    args = parse_args()

    if not args.root.is_dir():
        print(f"ERROR: root {args.root} does not exist or is unreadable",
              file=sys.stderr)
        sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    n_kept = 0
    n_total_rows = 0
    t0 = time.time()
    with args.out.open("w") as fh:
        for survey in args.surveys:
            for program in args.programs:
                print(f"[{survey}/{program}] scanning...")
                for k, (coadd, redrock, hpx) in enumerate(
                    walk_program(args.root, args.production, survey, program)
                ):
                    if k % args.hpix_stride != 0:
                        continue

                    n_rows = -1 if args.skip_row_count else count_rows(coadd)
                    rec = {
                        "coadd": str(coadd),
                        "redrock": str(redrock),
                        "survey": survey,
                        "program": program,
                        "healpix": hpx,
                        "n_rows": n_rows,
                    }
                    fh.write(json.dumps(rec) + "\n")
                    n_kept += 1
                    if n_rows > 0:
                        n_total_rows += n_rows

                    if n_kept % 100 == 0:
                        dt = time.time() - t0
                        print(f"  {n_kept} healpix indexed, "
                              f"~{n_total_rows} rows, {dt:.1f}s")

                    if args.max_healpix is not None and n_kept >= args.max_healpix:
                        print(f"  hit --max-healpix={args.max_healpix}, stopping")
                        break
                if args.max_healpix is not None and n_kept >= args.max_healpix:
                    break
            if args.max_healpix is not None and n_kept >= args.max_healpix:
                break

    print(f"\nDone. {n_kept} healpix files indexed -> {args.out}")
    if not args.skip_row_count:
        print(f"Approx total rows (pre-quality-cut): {n_total_rows}")


if __name__ == "__main__":
    main()
