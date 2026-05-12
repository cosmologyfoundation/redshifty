"""
sync_metrics_to_wandb
=====================
Replay a metrics.jsonl file into a wandb run.

Used to upload runs that completed BEFORE the wandb integration was
added (jobs 52827566 / 52827575 / earlier tokenizer trial), so their
loss curves show up in the wandb dashboard alongside live runs for
side-by-side comparison.

Handles both schemas:
- Old transformer metrics: {kind: train|val, step, loss, overall_acc,
  redshift_acc, spectrum_acc, ...}
- New transformer metrics: same + {loss_redshift, loss_spectrum, loss_total}
- Tokenizer metrics: {kind: train|val, step, loss_total, loss_recon,
  loss_quant, lr, ...}

Logs into the `redshifty` project (override with --project). Tags the
run so it's easy to filter on the dashboard.

Usage:
    # Single run
    python nersc/sync_metrics_to_wandb.py \\
        --metrics $SCRATCH/deepsrch/checkpoints/approach_a_52827566/metrics.jsonl \\
        --run-name approach_a_52827566_replay \\
        --tags pre-fix approach_a unweighted

    # Multiple runs at once
    for p in $SCRATCH/deepsrch/checkpoints/approach_*_52827*/metrics.jsonl; do
        run=$(basename $(dirname $p))_replay
        python nersc/sync_metrics_to_wandb.py --metrics $p --run-name $run \\
            --tags pre-fix unweighted
    done

Run from a login node; offline mode is also supported via --mode offline,
then `wandb sync` from a login node.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from src.training.wandb_util import init_wandb, wfinish, wlog  # noqa: E402


# Per-key namespacing. Anything starting with "val_" goes to val/<name>;
# everything else in a train record goes to train/<name>.
TRAIN_INCLUDE = {
    "loss", "loss_redshift", "loss_spectrum", "loss_total",
    "loss_recon", "loss_quant",  # tokenizer-flavor
    "overall_acc", "redshift_acc", "spectrum_acc",
    "lr", "steps_per_sec",
}
DROP = {"kind", "step", "timestamp", "elapsed_s"}


def parse_args():
    p = argparse.ArgumentParser(description="Replay metrics.jsonl into a wandb run")
    p.add_argument("--metrics", type=Path, required=True,
                   help="Path to metrics.jsonl from a finished run")
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--project", type=str, default="redshifty")
    p.add_argument("--mode", choices=["online", "offline", "disabled"],
                   default="online")
    p.add_argument("--tags", nargs="*", default=[],
                   help="Wandb tags for filtering (space-separated)")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Where wandb stores local files. Defaults to "
                        "<metrics-dir>/wandb-replay")
    p.add_argument("--config-json", type=Path, default=None,
                   help="Optional config.json from the run dir to log as "
                        "wandb config (default: look next to metrics file)")
    return p.parse_args()


def load_config(args) -> dict:
    if args.config_json is not None and args.config_json.is_file():
        return json.loads(args.config_json.read_text())
    candidate = args.metrics.parent / "config.json"
    if candidate.is_file():
        return json.loads(candidate.read_text())
    return {}


def main():
    args = parse_args()

    if not args.metrics.is_file():
        print(f"ERROR: {args.metrics} not found", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out_dir or (args.metrics.parent / "wandb-replay")

    config = load_config(args)
    config.update({
        "replay_source": str(args.metrics),
        "replay_tags": args.tags,
    })

    run = init_wandb(
        mode=args.mode,
        project=args.project,
        run_name=args.run_name,
        config=config,
        out_dir=out_dir,
    )
    if run is None:
        print(f"[replay] wandb disabled; aborting (mode={args.mode})")
        sys.exit(1)

    # Apply tags after init (wandb.init doesn't take tags via this helper)
    try:
        if args.tags:
            run.tags = list(args.tags)
    except Exception:
        pass

    n_train = n_val = 0
    last_step = -1

    with args.metrics.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  skip malformed line: {e}", file=sys.stderr)
                continue

            kind = rec.get("kind", "train")
            step = int(rec.get("step", last_step + 1))
            last_step = step

            payload = {}
            if kind == "val":
                # val records prefix every metric with "val_" already
                for k, v in rec.items():
                    if k in DROP:
                        continue
                    name = k[4:] if k.startswith("val_") else k
                    payload[f"val/{name}"] = v
                n_val += 1
            else:
                for k, v in rec.items():
                    if k in DROP:
                        continue
                    if k in TRAIN_INCLUDE:
                        payload[f"train/{k}"] = v
                n_train += 1

            if payload:
                wlog(run, payload, step=step)

    print(f"[replay] logged {n_train} train records, {n_val} val records, "
          f"last_step={last_step}")
    wfinish(run)


if __name__ == "__main__":
    main()
