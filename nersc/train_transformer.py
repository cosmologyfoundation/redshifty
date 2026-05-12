#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train SpectrumTransformer on DR1 with a pretrained tokenizer.

Thin CLI wrapper. All non-NERSC-specific training logic lives in
`src/training/` so it's reusable from notebooks / tests / `scripts/`
without dragging in DR1 paths. This file only handles:
  - argparse + smoke override
  - DR1 manifest loading (NERSC-specific filesystem paths)
  - SpectrumTokenizer load + RedshiftTokenizer fit
  - Healpix-level train/val partitioning of records
  - The actual training loop using helpers from src/training/

Usage (from inside a SLURM job; see train_transformer.slurm):

    python nersc/train_transformer.py \\
        --manifest $SCRATCH/deepsrch/manifests/dr1_smoke.jsonl \\
        --tokenizer-ckpt $SCRATCH/deepsrch/checkpoints/<run>/best.pt \\
        --approach a \\
        --steps 50000 \\
        --batch-size 8 \\
        --amp
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
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

# Core logic — moved to src/training/ for reusability.
from src.models.transformer import SpectrumTransformer  # noqa: E402
from src.tokenizers.redshift import RedshiftTokenizer  # noqa: E402
from src.tokenizers.spectrum import SpectrumTokenizer  # noqa: E402
from src.training.data_split import split_records_by_healpix  # noqa: E402
from src.training.eval import evaluate, evaluate_ar  # noqa: E402
from src.training.sequences import lr_at, tokenize_and_build  # noqa: E402
from src.training.utils import (  # noqa: E402
    compute_loss_breakdown,
    compute_masked_metrics,
    compute_metrics,
)
from src.training.wandb_util import init_wandb, log_model_artifact, wfinish, wlog  # noqa: E402

# NERSC-specific DR1 loaders.
from dr1_dataset import (  # noqa: E402
    DR1IndexedDataset,
    collate_dr1_skip_none,
    load_manifest,
)
from dr1_tokenized_dataset import collect_redshifts  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Train SpectrumTransformer on DR1")
    # Data
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--tokenizer-ckpt", type=Path, required=True,
                   help="Pretrained SpectrumTokenizer .pt (best.pt or final.pt)")
    p.add_argument("--max-spectra", type=int, default=None)
    p.add_argument("--healpix-holdout-frac", type=float, default=0.05,
                   help="Fraction of HEALPIX FILES (not rows) to reserve "
                        "for validation. Avoids same-pointing leakage.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--z-fit-files", type=int, default=200,
                   help="How many redrock files to scan when fitting RedshiftTokenizer")

    # Approach
    p.add_argument("--approach", choices=["a", "b"], required=True)

    # Model
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--n-encoder-layers", type=int, default=6)
    p.add_argument("--n-decoder-layers", type=int, default=6)
    p.add_argument("--n-heads", type=int, default=12)
    p.add_argument("--dropout", type=float, default=0.1)

    # Optim
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--redshift-loss-weight", type=float, default=50.0,
                   help="Multiplier on position-0 (redshift) loss term.")
    p.add_argument("--encoder-mask-ratio", type=float, default=0.15,
                   help="Fraction of encoder spectrum positions to replace "
                        "with [MASK]. BERT-style. 0.0 disables; 0.15 is "
                        "BERT canonical. Forces honest spectrum reconstruction.")
    p.add_argument("--ar-eval-batches", type=int, default=4,
                   help="Number of batches to run through autoregressive "
                        "eval at end-of-run and on best checkpoint.")

    # Logging
    p.add_argument("--run-name", type=str, default="approach_a")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"],
                   default="online")
    p.add_argument("--wandb-project", type=str, default="redshifty")
    p.add_argument("--push-wandb-artifact", action="store_true", default=True,
                   help="Upload a slim model-only best.pt to wandb as an Artifact "
                        "on each best update (and final). Disable with --no-push-wandb-artifact.")
    p.add_argument("--no-push-wandb-artifact", action="store_false",
                   dest="push_wandb_artifact")
    p.add_argument("--scratch-out", type=Path,
                   default=Path(os.environ.get("SCRATCH", "/tmp")) / "deepsrch")
    p.add_argument("--cfs-out", type=Path, default=None)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=2000)

    p.add_argument("--smoke", action="store_true",
                   help="Tiny config: 100 steps, 200 spectra, smaller model")
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.steps = 100
        args.max_spectra = 200
        args.batch_size = min(args.batch_size, 4)
        args.val_every = 50
        args.save_every = 100
        args.log_every = 10
        args.num_workers = 0
        args.warmup = 20
        args.d_model = 256
        args.n_encoder_layers = 2
        args.n_decoder_layers = 2
        args.n_heads = 8
        args.z_fit_files = min(args.z_fit_files, 5)
        args.ar_eval_batches = 1

    is_distributed = "RANK" in os.environ
    if is_distributed:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[setup] rank={rank}/{world_size} device={device} approach={args.approach} "
          f"steps={args.steps} mask_ratio={args.encoder_mask_ratio} "
          f"redshift_weight={args.redshift_loss_weight}")
    run_dir = args.scratch_out / "checkpoints" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    with (run_dir / "config.json").open("w") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v
                   for k, v in vars(args).items()}, f, indent=2)

    # Manifest
    print(f"[data] loading manifest {args.manifest}")
    records = load_manifest(args.manifest)
    print(f"[data] {len(records)} healpix records")

    # Healpix-level train/val split — no same-pointing leakage.
    train_records, val_records = split_records_by_healpix(
        records, holdout_frac=args.healpix_holdout_frac, seed=args.seed,
    )
    print(f"[data] healpix split: {len(train_records)} train, "
          f"{len(val_records)} val (frac={args.healpix_holdout_frac})")

    # Pretrained spectrum tokenizer (lives on GPU in main process; never forked)
    print(f"[tok] loading spectrum tokenizer {args.tokenizer_ckpt}")
    spec_tok = SpectrumTokenizer().to(device)
    ckpt = torch.load(args.tokenizer_ckpt, map_location=device, weights_only=False)
    sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    spec_tok.load_state_dict(sd)
    spec_tok.eval()
    for p in spec_tok.parameters():
        p.requires_grad_(False)

    # Fit redshift tokenizer on a sample of TRAIN manifest redshifts
    print(f"[tok] fitting redshift tokenizer on up to {args.z_fit_files} redrock files")
    zs = collect_redshifts(train_records, max_files=args.z_fit_files)
    print(f"[tok]   gathered {len(zs)} z values, min={zs.min():.4f} max={zs.max():.4f}")
    z_tok = RedshiftTokenizer(n_levels=256)
    z_tok.fit(zs)

    # Build separate datasets for each partition.
    train_ds = DR1IndexedDataset(
        train_records,
        require_good_zwarn=True,
        require_nonzero_flux=True,
        max_spectra=args.max_spectra,
    )
    val_ds = DR1IndexedDataset(
        val_records,
        require_good_zwarn=True,
        require_nonzero_flux=True,
        max_spectra=None if args.max_spectra is None else max(50, args.max_spectra // 10),
    )
    print(f"[data] train_ds={len(train_ds)} val_ds={len(val_ds)}")

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

    # Model
    model = SpectrumTransformer(
        d_model=args.d_model,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    ).to(device)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params:,} (~{n_params/1e6:.1f}M)")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    # Wandb init — rank 0 only
    wandb_run = None
    if rank == 0:
        wandb_config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        wandb_config.update({
            "n_params": n_params,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "n_manifest_records": len(records),
            "n_train_records": len(train_records),
            "n_val_records": len(val_records),
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
    best_val = float("inf")
    t0 = time.time()
    train_iter = iter(train_loader)
    model.train()
    if train_sampler is not None:
        train_sampler.set_epoch(0)

    while step < args.steps:
        try:
            raw = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            raw = next(train_iter)
        if raw is None:
            continue

        for g_ in optim.param_groups:
            g_["lr"] = lr_at(step, args.lr, args.warmup, args.steps)

        enc, dec, tgt, mask_pos = tokenize_and_build(
            raw, spec_tok, z_tok, args.approach, device,
            encoder_mask_ratio=args.encoder_mask_ratio,
        )

        optim.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=args.amp):
            logits, loss = model(enc, dec, targets=tgt,
                                 redshift_weight=args.redshift_loss_weight)
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optim)
        scaler.update()

        if rank == 0 and step % args.log_every == 0:
            with torch.no_grad():
                m = compute_metrics(logits, tgt)
                b = compute_loss_breakdown(logits, tgt)
                mm = (compute_masked_metrics(logits, tgt, mask_pos)
                      if mask_pos is not None else
                      {"masked_spec_acc": float("nan"), "n_masked": 0})
            dt = time.time() - t0
            rate = (step + 1) / max(dt, 1e-6)
            msg = {
                "kind": "train", "step": step, "lr": optim.param_groups[0]["lr"],
                "loss": float(loss.item()),
                **m, **b,
                "masked_spec_acc": mm["masked_spec_acc"],
                "n_masked": mm["n_masked"],
                "steps_per_sec": rate, "elapsed_s": dt,
            }
            print(f"[step {step:6d}] loss={msg['loss']:.4f} "
                  f"z_loss={b['loss_redshift']:.3f} spec_loss={b['loss_spectrum']:.3f} "
                  f"z_acc={m['redshift_acc']:.3f} spec_acc={m['spectrum_acc']:.3f} "
                  f"masked_acc={mm['masked_spec_acc']:.3f} "
                  f"{rate:.1f} step/s")
            with metrics_path.open("a") as f:
                f.write(json.dumps(msg) + "\n")
            wlog(wandb_run, {
                "train/loss": msg["loss"],
                "train/loss_redshift": b["loss_redshift"],
                "train/loss_spectrum": b["loss_spectrum"],
                "train/loss_total_unweighted": b["loss_total"],
                "train/lr": msg["lr"],
                "train/overall_acc": m["overall_acc"],
                "train/redshift_acc": m["redshift_acc"],
                "train/spectrum_acc": m["spectrum_acc"],
                "train/masked_spec_acc": mm["masked_spec_acc"],
                "train/steps_per_sec": rate,
            }, step=step)

        if rank == 0 and step > 0 and step % args.val_every == 0:
            v = evaluate(
                model, val_loader, spec_tok, z_tok, args.approach, device,
                args.amp, args.redshift_loss_weight,
                encoder_mask_ratio=args.encoder_mask_ratio,
            )
            model.train()
            print(f"[val   {step:6d}] " + " ".join(f"{k}={v[k]:.4f}" for k in v))
            with metrics_path.open("a") as f:
                f.write(json.dumps({"kind": "val", "step": step,
                                    **{f"val_{k}": vv for k, vv in v.items()}}) + "\n")
            wlog(wandb_run, {f"val/{k}": vv for k, vv in v.items()}, step=step)
            if v["loss"] < best_val:
                best_val = v["loss"]
                p = run_dir / "best.pt"
                torch.save({
                    "step": step,
                    "model": model.module.state_dict() if is_distributed else model.state_dict(),
                    "optim": optim.state_dict(),
                    "scaler": scaler.state_dict(),
                    "val_loss": best_val,
                    "z_tokenizer": {
                        "sorted_z": z_tok._sorted_z.cpu(),
                        "n_levels": z_tok.n_levels,
                        "gaussian_range": z_tok.gaussian_range,
                    },
                    "tokenizer_ckpt_path": str(args.tokenizer_ckpt),
                    "approach": args.approach,
                    "encoder_mask_ratio": args.encoder_mask_ratio,
                }, p)
                print(f"  *** new best val_loss={best_val:.4f} -> {p}")
                if args.cfs_out is not None:
                    try:
                        args.cfs_out.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(p, args.cfs_out / "best.pt")
                    except OSError as e:
                        print(f"  WARN: cfs_out mirror failed ({e}); "
                              f"SCRATCH best.pt is safe at {p}")

                if args.push_wandb_artifact and wandb_run is not None:
                    log_model_artifact(
                        wandb_run, p,
                        name=f"approach_{args.approach}_{args.run_name}",
                        aliases=["best", f"step_{step}"],
                        metadata={
                            "val_loss": best_val,
                            "step": step,
                            "approach": args.approach,
                            "encoder_mask_ratio": args.encoder_mask_ratio,
                            "redshift_loss_weight": args.redshift_loss_weight,
                            "full_state": True,
                        },
                    )

                ar = evaluate_ar(
                    model, val_loader, spec_tok, z_tok, args.approach, device,
                    max_batches=args.ar_eval_batches,
                    encoder_mask_ratio=args.encoder_mask_ratio,
                )
                model.train()
                print(f"  [ar_best {step:6d}] " + " ".join(f"{k}={ar[k]}" for k in ar))
                with metrics_path.open("a") as f:
                    f.write(json.dumps({"kind": "ar_best", "step": step,
                                        **{f"val_ar_{k}": vv for k, vv in ar.items()}}) + "\n")
                wlog(wandb_run, {f"val_ar/{k}": vv for k, vv in ar.items()}, step=step)

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
            "z_tokenizer": {
                "sorted_z": z_tok._sorted_z.cpu(),
                "n_levels": z_tok.n_levels,
                "gaussian_range": z_tok.gaussian_range,
            },
            "tokenizer_ckpt_path": str(args.tokenizer_ckpt),
            "approach": args.approach,
            "encoder_mask_ratio": args.encoder_mask_ratio,
        }, p)
        print(f"[done] final -> {p}  best_val_loss={best_val:.4f}")
        if args.cfs_out is not None:
            try:
                args.cfs_out.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, args.cfs_out / "final.pt")
            except OSError as e:
                print(f"  WARN: cfs_out final mirror failed ({e}); "
                      f"SCRATCH final.pt is safe at {p}")

        if args.push_wandb_artifact and wandb_run is not None:
            log_model_artifact(
                wandb_run, p,
                name=f"approach_{args.approach}_{args.run_name}",
                aliases=["final", f"step_{step}"],
                metadata={"step": step, "approach": args.approach},
            )

        ar_final = evaluate_ar(
            model, val_loader, spec_tok, z_tok, args.approach, device,
            max_batches=args.ar_eval_batches,
            encoder_mask_ratio=args.encoder_mask_ratio,
        )
        print(f"[ar_final {step:6d}] " + " ".join(f"{k}={ar_final[k]}" for k in ar_final))
        with metrics_path.open("a") as f:
            f.write(json.dumps({"kind": "ar_final", "step": step,
                                **{f"val_ar_{k}": vv for k, vv in ar_final.items()}}) + "\n")
        wlog(wandb_run, {f"val_ar/{k}": vv for k, vv in ar_final.items()}, step=step)

        wfinish(wandb_run)

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
