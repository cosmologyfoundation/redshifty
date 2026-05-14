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
from src.models.transformer import SpectrumTransformer, MASK_TOKEN, TOTAL_VOCAB_SIZE  # noqa: E402
from src.tokenizers.redshift import RedshiftTokenizer  # noqa: E402
from src.tokenizers.redshift_v2 import RedshiftTokenizerV2  # noqa: E402
from src.tokenizers.spectrum import SpectrumTokenizer  # noqa: E402
from src.tokenizers.spectrum_v2 import SpectrumTokenizerV2  # noqa: E402
from src.training.data_split import split_records_by_healpix  # noqa: E402
from src.training.eval import evaluate, evaluate_ar  # noqa: E402
from src.training.sequences import lr_at, tokenize_and_build  # noqa: E402
from src.training.utils import (  # noqa: E402
    compute_all_auc,
    compute_all_r2,
    compute_loss_breakdown,
    compute_masked_auc,
    compute_masked_metrics,
    compute_masked_redshift_acc,
    compute_masked_r2,
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
    p.add_argument("--tokenizer-kind", choices=["v1", "v2"], default="v1",
                   help="Spectrum tokenizer version: v1 (ConvNeXt-V2 + LFQ, val_recon=1.35) "
                        "or v2 (U-Net + cross-attn + entropy loss, val_recon=0.157). "
                        "Default v1.")
    p.add_argument("--redshift-levels", type=int, default=256,
                   help="Number of FSQ levels for RedshiftTokenizer. "
                        "V1 default 256. V2 default 1024.")
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
    p.add_argument("--decoder-vocab-size", type=int, default=2056,
                   help="Decoder vocabulary size. Must be >= redshift token count. "
                        "Default 2056. Use smaller value if decoder generates only redshift "
                        "(e.g., 262 = 6 special + 256 redshift). Larger matches encoder size.")
    p.add_argument("--decoder-corrupt-ratio", type=float, default=0.0,
                   help="Fraction of decoder positions to corrupt with [MASK] during training. "
                        "BERT-style corruption on decoder input. 0.0 = no corruption (teacher forcing). "
                        "0.15-0.30 recommended for denoising objective. Forces decoder to generate "
                        "rather than rely on teacher-forced input.")

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
    p.add_argument("--encoder-mask-ratio", type=float, default=0.50,
                   help="Fraction of encoder spectrum positions to replace "
                        "with [MASK]. BERT-style. 0.0 disables; 0.50 recommended "
                        "for copy prevention. Forces honest spectrum reconstruction.")
    p.add_argument("--ar-eval-batches", type=int, default=4,
                   help="Number of batches to run through autoregressive "
                        "eval at end-of-run and on best checkpoint.")
    p.add_argument("--ar-train-ratio", type=float, default=0.0,
                   help="Fraction of training steps that use full autoregressive "
                        "loss instead of teacher forcing. 0.0 = pure teacher forcing. "
                        "Schedules from 0 to this value after --ar-train-start steps. "
                        "Helps close the train/val gap by training on model own predictions.")
    p.add_argument("--ar-train-start", type=int, default=5000,
                   help="Start AR training after this many steps (allow encoder to learn first).")
    p.add_argument("--ar-max-tokens", type=int, default=50,
                   help="Max tokens to generate during AR training (truncated for speed).")

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

    # Detect DDP: either explicit RANK env var (from SLURM script exports)
    # or SLURM's native SLURM_PROCID (from interactive srun).
    is_distributed = "RANK" in os.environ or "SLURM_PROCID" in os.environ
    if is_distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
        rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
        world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
        # PyTorch env:// rendezvous requires RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        # When srun uses --gpus-per-task=N, each task sees only its
        # bound GPU(s) as cuda:0.., so local_rank may exceed device_count.
        # When all GPUs are visible to every task, local_rank is the
        # actual device index. Pick whichever applies.
        cuda_idx = local_rank if local_rank < torch.cuda.device_count() else 0
        torch.cuda.set_device(cuda_idx)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{cuda_idx}")
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
    print(f"[tok] loading spectrum tokenizer {args.tokenizer_ckpt} (kind={args.tokenizer_kind})")
    if args.tokenizer_kind == "v2":
        spec_tok = SpectrumTokenizerV2().to(device)
    else:
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
    if args.tokenizer_kind == "v2":
        z_tok = RedshiftTokenizerV2(n_levels=1024, d_model=args.d_model)
    else:
        z_tok = RedshiftTokenizer(n_levels=args.redshift_levels)
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

    # Model — partially decoupled vocabularies (Option 2)
    model = SpectrumTransformer(
        vocab_size=TOTAL_VOCAB_SIZE,
        d_model=args.d_model,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    ).to(device)
    if is_distributed:
        model = DDP(model, device_ids=[device.index], find_unused_parameters=False)
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
    epoch = 0
    best_val = float("inf")
    t0 = time.time()
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    train_iter = iter(train_loader)
    model.train()

    while step < args.steps:
        try:
            raw = next(train_iter)
        except StopIteration:
            epoch += 1
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_iter = iter(train_loader)
            raw = next(train_iter)
        if raw is None:
            continue

        for g_ in optim.param_groups:
            g_["lr"] = lr_at(step, args.lr, args.warmup, args.steps)

        enc, dec, tgt, mask_pos, rz_mask = tokenize_and_build(
            raw, spec_tok, z_tok, args.approach, device,
            encoder_mask_ratio=args.encoder_mask_ratio,
        )

        # Decoder corruption (BERT-style): replace random decoder positions with MASK
        # and compute loss ONLY on corrupted positions (not teacher-forced)
        # This forces decoder to generate from cross-attention features, not copy
        if args.decoder_corrupt_ratio > 0.0:
            dec_corrupt_mask = torch.rand_like(dec.float()) < args.decoder_corrupt_ratio
            dec_input_corrupted = dec.clone()
            dec_input_corrupted[dec_corrupt_mask] = MASK_TOKEN
            dec_for_loss = dec_input_corrupted
            tgt_for_loss = tgt.clone()
            tgt_for_loss[~dec_corrupt_mask] = -100
        else:
            dec_for_loss = dec
            tgt_for_loss = tgt

        is_ar_step = (
            args.ar_train_ratio > 0 and
            step >= args.ar_train_start and
            step % 100 < int(args.ar_train_ratio * 100)
        )

        optim.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=args.amp):
            if is_ar_step:
                logits, loss = model.ar_loss(
                    enc, tgt,
                    max_generate_tokens=args.ar_max_tokens,
                    redshift_weight=args.redshift_loss_weight,
                )
            else:
                logits, loss = model(enc, dec_for_loss, targets=tgt_for_loss,
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
                rm = (compute_masked_redshift_acc(logits, tgt, rz_mask)
                      if rz_mask is not None else
                      {"redshift_acc_masked": float("nan"), "n_rz_masked": 0})
                ma = (compute_masked_auc(logits, tgt, mask_pos)
                      if mask_pos is not None else
                      {"mean_mask_auc": float("nan"), "n_masked": 0})
                mr = (compute_masked_r2(logits, tgt, mask_pos)
                      if mask_pos is not None else
                      {"masked_spec_r2": float("nan"), "n_masked": 0})
                aa = compute_all_auc(logits, tgt)
                ar = compute_all_r2(logits, tgt)
            dt = time.time() - t0
            rate = (step + 1) / max(dt, 1e-6)
            msg = {
                "kind": "train", "step": step, "lr": optim.param_groups[0]["lr"],
                "loss": float(loss.item()),
                "is_ar": is_ar_step,
                **m, **b,
                "masked_spec_acc": mm["masked_spec_acc"],
                "n_masked": mm["n_masked"],
                "redshift_acc_masked": rm["redshift_acc_masked"],
                "n_rz_masked": rm["n_rz_masked"],
                "mean_mask_auc": ma["mean_mask_auc"],
                "masked_spec_r2": mr["masked_spec_r2"],
                "all_mean_auc": aa["all_mean_auc"],
                "all_spec_r2": ar["all_spec_r2"],
                "steps_per_sec": rate, "elapsed_s": dt,
            }
            ar_tag = " [AR]" if is_ar_step else ""
            print(f"[step {step:6d}{ar_tag}] loss={msg['loss']:.4f} "
                  f"z_loss={b['loss_redshift']:.3f} spec_loss={b['loss_spectrum']:.3f} "
                  f"z_acc={m['redshift_acc']:.3f} spec_acc={m['spectrum_acc']:.3f} "
                  f"masked_acc={mm['masked_spec_acc']:.3f} "
                  f"rz_masked_acc={rm['redshift_acc_masked']:.3f} "
                  f"mask_r2={mr['masked_spec_r2']:.3f} mask_auc={ma['mean_mask_auc']:.3f} "
                  f"all_r2={ar['all_spec_r2']:.3f} all_auc={aa['all_mean_auc']:.3f} "
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
                "train/redshift_acc_masked": rm["redshift_acc_masked"],
                "train/mean_mask_auc": ma["mean_mask_auc"],
                "train/masked_spec_r2": mr["masked_spec_r2"],
                "train/all_mean_auc": aa["all_mean_auc"],
                "train/all_spec_r2": ar["all_spec_r2"],
                "train/steps_per_sec": rate,
            }, step=step)

        if rank == 0 and step > 0 and step % args.val_every == 0:
            v = evaluate(
                model.module if is_distributed else model,
                val_loader, spec_tok, z_tok, args.approach, device,
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
                    model.module if is_distributed else model,
                    val_loader, spec_tok, z_tok, args.approach, device,
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
            model.module if is_distributed else model,
            val_loader, spec_tok, z_tok, args.approach, device,
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
