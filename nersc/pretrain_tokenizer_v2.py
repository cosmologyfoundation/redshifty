"""
Pretrain the spectrum tokenizer V2 on DR1 spectra.

Stage 1: Stabilize training with:
  - Lower commitment weight (0.05 vs V1's 0.25)
  - Codebook entropy loss
  - Top-hat 5-pixel smoothing
  - Lower lr (1e-4) with longer warmup (1000 steps)

Inputs:
  - JSONL manifest from build_dr1_index.py
  - 1000 healpix ~2M spectra target
Outputs:
  - Periodic checkpoints to $SCRATCH/<run_name>/
  - Best-val checkpoint mirrored to $CFS_OUT
  - metrics.jsonl with per-step train + per-epoch val records

Single-GPU AMP loop with optional DDP.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from src.tokenizers.spectrum_v2 import SpectrumTokenizerV2
from src.training.wandb_util import init_wandb, log_model_artifact, wfinish, wlog

sys.path.insert(0, str(HERE))
from dr1_dataset import (
    DR1IndexedDataset,
    collate_dr1_skip_none,
    load_manifest,
)


def parse_args():
    p = argparse.ArgumentParser(description="Pretrain DESI spectrum tokenizer V2")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--max-spectra", type=int, default=None)
    p.add_argument("--val-frac", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--amp", action="store_true")

    p.add_argument("--run-name", type=str, default="tokenizer_v2_s1")
    p.add_argument("--scratch-out", type=Path,
                   default=Path(os.environ.get("SCRATCH", "/tmp")) / "deepsrch")
    p.add_argument("--cfs-out", type=Path, default=None)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"],
                   default="offline")
    p.add_argument("--wandb-project", type=str, default="redshifty")
    p.add_argument("--push-wandb-artifact", action="store_true", default=True)
    p.add_argument("--no-push-wandb-artifact", action="store_false",
                   dest="push_wandb_artifact")

    p.add_argument("--entropy-weight", type=float, default=0.1,
                   help="Weight for codebook entropy loss")
    p.add_argument("--commitment-weight", type=float, default=0.05,
                   help="Commitment loss weight (V1 used 0.25)")
    p.add_argument("--use-tophat", action="store_true", default=True,
                   help="Enable top-hat smoothing preprocessing")
    p.add_argument("--no-tophat", action="store_false", dest="use_tophat",
                   help="Disable top-hat smoothing")
    p.add_argument("--use-skip", action="store_true", default=True)
    p.add_argument("--no-skip", action="store_false", dest="use_skip")
    p.add_argument("--use-cross-attention", action="store_true", default=True)
    p.add_argument("--no-cross-attention", action="store_false", dest="use_cross_attention")

    p.add_argument("--smoke", action="store_true",
                   help="Tiny config: 50 steps, 200 spectra, no AMP")
    return p.parse_args()


def lr_at(step: int, base_lr: float, warmup: int, total: int) -> float:
    import math
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def evaluate(model, loader, device, amp: bool, max_batches: int = 50):
    model.eval()
    losses = {"total": 0.0, "recon": 0.0, "quant": 0.0, "commit": 0.0, "entropy": 0.0}
    n = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if batch is None:
                continue
            if i >= max_batches:
                break
            flux = batch["flux"].to(device, non_blocking=True)
            ivar = batch["ivar"].to(device, non_blocking=True)
            istd = torch.sqrt(ivar.clamp(min=1e-10))
            x = torch.stack([flux, istd], dim=1)
            with torch.amp.autocast("cuda", enabled=amp):
                _, loss, _ = model(x)
            for k in losses:
                losses[k] += loss[k].item()
            n += 1
    if n == 0:
        return {k: float("nan") for k in losses}
    return {k: v / n for k, v in losses.items()}


def main():
    args = parse_args()
    if args.smoke:
        args.steps = 50
        args.max_spectra = 200
        args.val_every = 25
        args.save_every = 50
        args.log_every = 5
        args.batch_size = min(args.batch_size, 4)
        args.num_workers = 0

    is_distributed = "RANK" in os.environ or "SLURM_PROCID" in os.environ
    if is_distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
        rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
        world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        cuda_idx = local_rank if local_rank < torch.cuda.device_count() else 0
        torch.cuda.set_device(cuda_idx)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{cuda_idx}")
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[setup] rank={rank}/{world_size} device={device} amp={args.amp} steps={args.steps}")
    print(f"[setup] scratch_out={args.scratch_out}")
    print(f"[setup] entropy_weight={args.entropy_weight} commitment_weight={args.commitment_weight}")
    print(f"[setup] tophat={args.use_tophat} skip={args.use_skip} ca={args.use_cross_attention}")

    run_dir = args.scratch_out / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

    with (run_dir / "config.json").open("w") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v
                   for k, v in vars(args).items()}, f, indent=2)

    print(f"[data] loading manifest {args.manifest}")
    records = load_manifest(args.manifest)
    print(f"[data] {len(records)} healpix records")

    full = DR1IndexedDataset(
        records,
        require_good_zwarn=True,
        require_nonzero_flux=True,
        max_spectra=args.max_spectra,
    )
    print(f"[data] {len(full)} spectra in flat index")

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(full), generator=g).tolist()
    n_val = max(1, int(len(full) * args.val_frac))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    train_ds = Subset(full, train_idx)
    val_ds = Subset(full, val_idx)
    print(f"[data] train={len(train_ds)} val={len(val_ds)}")

    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_distributed else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_dr1_skip_none,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, args.num_workers // 2),
        collate_fn=collate_dr1_skip_none,
        pin_memory=device.type == "cuda",
    )

    model = SpectrumTokenizerV2(
        commitment_weight=args.commitment_weight,
        entropy_weight=args.entropy_weight,
        use_tophat=args.use_tophat,
        use_skip_connections=args.use_skip,
        use_cross_attention=args.use_cross_attention,
    ).to(device)
    if is_distributed:
        model = DDP(model, device_ids=[device.index], find_unused_parameters=False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params:,} (~{n_params/1e6:.1f}M) V2")

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    wandb_run = None
    if rank == 0:
        wandb_config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        wandb_config.update({
            "n_params": n_params,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
        })
        wandb_dir = args.scratch_out / "wandb" / args.run_name
        wandb_run = init_wandb(
            mode=args.wandb_mode,
            project=args.wandb_project,
            run_name=args.run_name,
            config=wandb_config,
            out_dir=wandb_dir,
        )

    step = 0
    epoch = 0
    best_val = float("inf")
    t_start = time.time()
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    train_iter = iter(train_loader)
    model.train()

    while step < args.steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            epoch += 1
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_iter = iter(train_loader)
            batch = next(train_iter)
        if batch is None:
            continue

        for g_ in optim.param_groups:
            g_["lr"] = lr_at(step, args.lr, args.warmup, args.steps)

        flux = batch["flux"].to(device, non_blocking=True)
        ivar = batch["ivar"].to(device, non_blocking=True)
        istd = torch.sqrt(ivar.clamp(min=1e-10))
        x = torch.stack([flux, istd], dim=1)

        optim.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=args.amp):
            _, loss, _ = model(x)

        scaler.scale(loss["total"]).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optim)
        scaler.update()

        if rank == 0 and step % args.log_every == 0:
            dt = time.time() - t_start
            rate = (step + 1) / max(dt, 1e-6)
            msg = {
                "kind": "train",
                "step": step,
                "lr": optim.param_groups[0]["lr"],
                "loss_total": float(loss["total"].item()),
                "loss_recon": float(loss["recon"].item()),
                "loss_quant": float(loss["quant"].item()),
                "loss_commit": float(loss["commit"].item()),
                "loss_entropy": float(loss["entropy"].item()),
                "steps_per_sec": rate,
                "elapsed_s": dt,
            }
            print(
                f"[step {step:6d}] "
                f"loss={msg['loss_total']:.4f} "
                f"(recon={msg['loss_recon']:.4f}, quant={msg['loss_quant']:.4f} "
                f"commit={msg['loss_commit']:.4f}, ent={msg['loss_entropy']:.4f}) "
                f"lr={msg['lr']:.2e} {rate:.1f} step/s"
            )
            with metrics_path.open("a") as f:
                f.write(json.dumps(msg) + "\n")
            wlog(wandb_run, {
                "train/loss_total": msg["loss_total"],
                "train/loss_recon": msg["loss_recon"],
                "train/loss_quant": msg["loss_quant"],
                "train/loss_commit": msg["loss_commit"],
                "train/loss_entropy": msg["loss_entropy"],
                "train/lr": msg["lr"],
                "train/steps_per_sec": rate,
            }, step=step)

        if rank == 0 and step > 0 and step % args.val_every == 0:
            val_losses = evaluate(
                model.module if is_distributed else model,
                val_loader, device, args.amp,
            )
            model.train()
            msg = {"kind": "val", "step": step, **{f"val_{k}": v for k, v in val_losses.items()}}
            print(f"[val   {step:6d}] " + " ".join(f"{k}={v:.4f}" for k, v in val_losses.items()))
            with metrics_path.open("a") as f:
                f.write(json.dumps(msg) + "\n")
            wlog(wandb_run, {f"val/{k}": v for k, v in val_losses.items()}, step=step)

            if val_losses["total"] < best_val:
                best_val = val_losses["total"]
                ckpt = {
                    "step": step,
                    "model": model.module.state_dict() if is_distributed else model.state_dict(),
                    "optim": optim.state_dict(),
                    "scaler": scaler.state_dict(),
                    "val_loss": best_val,
                    "args": vars(args) | {"manifest": str(args.manifest)},
                }
                p = run_dir / "best.pt"
                torch.save(ckpt, p)
                print(f"  *** new best val_loss={best_val:.4f} -> {p}")
                if args.cfs_out is not None:
                    try:
                        args.cfs_out.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(p, args.cfs_out / "best.pt")
                    except OSError as e:
                        print(f"  WARN: cfs_out mirror failed ({e})")

                if args.push_wandb_artifact and wandb_run is not None:
                    log_model_artifact(
                        wandb_run, p,
                        name=f"tokenizer_{args.run_name}",
                        aliases=["best", f"step_{step}"],
                        metadata={
                            "val_loss": best_val,
                            "step": step,
                            "full_state": True,
                        },
                    )

        if rank == 0 and step > 0 and step % args.save_every == 0:
            p = run_dir / f"step_{step:08d}.pt"
            torch.save({
                "step": step,
                "model": model.module.state_dict() if is_distributed else model.state_dict(),
            }, p)
            print(f"  ckpt -> {p}")

        step += 1

    if rank == 0:
        p = run_dir / "final.pt"
        torch.save({
            "step": step,
            "model": model.module.state_dict() if is_distributed else model.state_dict(),
        }, p)
        print(f"[done] final -> {p}")
        if args.cfs_out is not None:
            try:
                args.cfs_out.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, args.cfs_out / "final.pt")
            except OSError as e:
                print(f"  WARN: cfs_out mirror failed ({e})")
        print(f"[done] best val_loss={best_val:.4f}")

        if args.push_wandb_artifact and wandb_run is not None:
            log_model_artifact(
                wandb_run, p,
                name=f"tokenizer_{args.run_name}",
                aliases=["final", f"step_{step}"],
                metadata={"step": step},
            )

        wfinish(wandb_run)

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()