"""
Wandb helper shared by the NERSC trainers.

- Reads WANDB_API_KEY from a repo-root .env via python-dotenv.
- Supports online / offline / disabled modes.
- Logs to a stable subdir of $SCRATCH so artifacts survive the SLURM
  step but are easy to clean up.

All functions are no-ops when mode == "disabled", and tolerate wandb
import failures gracefully (helpful for compute nodes that can't reach
PyPI to upgrade the module). Tests on a CPU laptop work fine in
"disabled" mode without ever importing wandb.
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
        wandb Run object, or None if mode == "disabled".
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
        # python-dotenv isn't installed; rely on the env directly.
        pass

    try:
        import wandb
    except ImportError:
        print("[wandb] wandb not installed; logging disabled")
        return None

    if mode == "online" and not os.environ.get("WANDB_API_KEY"):
        print("[wandb] WANDB_API_KEY not set; falling back to offline mode")
        mode = "offline"

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
