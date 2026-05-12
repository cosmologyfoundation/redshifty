#!/usr/bin/env python3
"""
Upload a trained spectrum-tokenizer checkpoint to W&B as a model artifact.

The transformer trainer already pushes itself to W&B on every best-val
step. The tokenizer pretraining loop doesn't, since it's a one-time
artifact — easier to upload after the fact with this script.

Usage:
    python nersc/push_tokenizer_artifact.py \
        --ckpt $SCRATCH/deepsrch/checkpoints/tokenizer_v1_52693687/best.pt \
        --name spectrum_tokenizer_v1 \
        --alias best --alias v1

After upload the artifact URI is printed, e.g.:
    <entity>/redshifty/spectrum_tokenizer_v1:v0  (alias: best, v1)

Use that URI in the notebook's TOKENIZER_ARTIFACT_URI to pull it down.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from src.training.wandb_util import init_wandb, log_model_artifact, wfinish  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True,
                   help="Path to tokenizer .pt to upload")
    p.add_argument("--name", default="spectrum_tokenizer_v1",
                   help="Artifact name (versioned automatically)")
    p.add_argument("--alias", action="append", default=[],
                   help="Alias(es) to attach (repeatable). Default: ['best']")
    p.add_argument("--project", default="redshifty")
    p.add_argument("--run-name", default=None,
                   help="W&B run name (default: derived from --name)")
    p.add_argument("--scratch-out", type=Path,
                   default=Path("/tmp") / "wandb_uploader",
                   help="Local dir for wandb files (uploader is transient)")
    p.add_argument("--no-prune", action="store_true",
                   help="Keep prior artifact versions (default: prune)")
    return p.parse_args()


def main():
    args = parse_args()
    assert args.ckpt.is_file(), f"ckpt not found: {args.ckpt}"
    aliases = args.alias or ["best"]
    run_name = args.run_name or f"upload_{args.name}"

    run = init_wandb(
        mode="online",
        project=args.project,
        run_name=run_name,
        config={"source_path": str(args.ckpt), "artifact_name": args.name},
        out_dir=args.scratch_out,
    )
    if run is None:
        print("[upload] wandb not available; aborting")
        sys.exit(1)

    art = log_model_artifact(
        run, args.ckpt,
        name=args.name,
        aliases=aliases,
        metadata={"source_path": str(args.ckpt)},
        keep_only_latest=not args.no_prune,
    )
    if art is not None:
        art.wait()
        print(f"\n[upload] {run.entity}/{run.project}/{args.name}:{art.version}"
              f"  aliases={aliases}")
    wfinish(run)


if __name__ == "__main__":
    main()
