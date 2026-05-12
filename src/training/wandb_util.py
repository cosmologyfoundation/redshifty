"""
Wandb helper used by training scripts (NERSC + local).

- Reads `WANDB_API_KEY` from a repo-root `.env` via python-dotenv.
- Supports online / offline / disabled modes.
- Forces `os.environ["WANDB_MODE"]` before calling `wandb.init` so any
  upstream-set `WANDB_MODE` (e.g. NERSC's pytorch module defaulting to
  offline) doesn't silently override the CLI choice.

All functions are no-ops when `mode == "disabled"`, and tolerate
wandb / python-dotenv import failures gracefully so the test suite can
run on a CPU laptop without installing them.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def init_wandb(
    mode: str,
    project: str,
    run_name: str,
    config: dict,
    out_dir: Path,
    dotenv_path: Optional[Path] = None,
):
    """Initialize a wandb run (or return None if disabled).

    Args:
        mode: "online", "offline", or "disabled".
        project: wandb project name.
        run_name: human-readable run name.
        config: dict of hyperparameters / config to log.
        out_dir: directory for wandb's local files (relative or absolute).
            Typically `$SCRATCH/deepsrch/wandb/<run>`.
        dotenv_path: path to .env (default: search from cwd upward).

    Returns:
        wandb Run object, or None if mode == "disabled" or init fails.
    """
    if mode == "disabled":
        return None

    try:
        from dotenv import load_dotenv
        if dotenv_path is not None:
            load_dotenv(dotenv_path)
        else:
            load_dotenv()
    except ImportError:
        pass

    if mode == "online" and not os.environ.get("WANDB_API_KEY"):
        print("[wandb] WANDB_API_KEY not set; falling back to offline mode")
        mode = "offline"

    # CRITICAL: set env var BEFORE importing/initing wandb so that any
    # ambient WANDB_MODE (e.g. NERSC pytorch module sets it to "offline")
    # is overridden by our explicit choice.
    os.environ["WANDB_MODE"] = mode

    try:
        import wandb
    except ImportError:
        print("[wandb] wandb not installed; logging disabled")
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        run = wandb.init(
            project=project,
            name=run_name,
            mode=mode,
            config=config,
            dir=str(out_dir),
        )
        print(f"[wandb] mode={mode} project={project} name={run_name}")
        if hasattr(run, "url") and run.url:
            print(f"[wandb] url: {run.url}")
        return run
    except Exception as e:
        print(f"[wandb] init failed ({e!r}); continuing without wandb")
        return None


def wlog(run, payload: dict, step: int):
    """Log a metric dict (no-op if run is None)."""
    if run is None:
        return
    try:
        run.log(payload, step=step)
    except Exception as e:
        print(f"[wandb] log failed ({e!r})")


def wfinish(run):
    """Close the wandb run (no-op if None)."""
    if run is None:
        return
    try:
        run.finish()
    except Exception:
        pass


def log_model_artifact(
    run,
    ckpt_path,
    name: str,
    aliases=None,
    metadata=None,
):
    """Upload a checkpoint to wandb as a model artifact.

    Use sparingly — wandb has storage limits, so prefer slim model-only
    checkpoints (no optim/scaler state). For the redshifty project,
    best.pt with full optim state is ~300 MB; a slim model-only version
    is ~100 MB. Recommended pattern: write a slim copy alongside best.pt
    and log THAT.

    Args:
        run: wandb run from init_wandb (or None — no-op).
        ckpt_path: path to the .pt file to upload.
        name: artifact name (e.g. "approach_a_best"). Versioned
            automatically by wandb (v0, v1, ...).
        aliases: list of human-readable aliases to attach
            (e.g. ["best", "step_12000"]).
        metadata: optional dict of key=value pairs to attach.

    Returns:
        wandb.Artifact or None.
    """
    if run is None:
        return None
    try:
        import wandb
    except ImportError:
        return None
    try:
        art = wandb.Artifact(
            name=name,
            type="model",
            metadata=metadata or {},
        )
        art.add_file(str(ckpt_path))
        run.log_artifact(art, aliases=aliases or [])
        print(f"[wandb] logged artifact {name} <- {ckpt_path}")
        return art
    except Exception as e:
        print(f"[wandb] artifact upload failed ({e!r})")
        return None
