"""
Train SpectrumTransformer on DR1 with a pretrained tokenizer.

Drop-in NERSC counterpart to scripts/train.py:
- Reads DR1 healpix coadds via the manifest from build_dr1_index.py
- Loads pretrained SpectrumTokenizer weights (frozen, eval)
- Fits the redshift tokenizer on a sample of manifest redshifts
- Trains the transformer for either Approach A (joint) or B (masked)
- Single-GPU AMP loop; mirrors best/final checkpoints to $CFS_OUT

DataLoader workers produce raw spectra only (no CUDA touches). The
spectrum tokenizer runs on the full batch on the main process's GPU --
this avoids the "Cannot re-initialize CUDA in forked subprocess" error
and is much faster than per-item encode in workers.

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
import math
import os
import shutil
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from src.models.transformer import (  # noqa: E402
    EOS_TOKEN,
    REDSHIFT_TOKEN_OFFSET,
    SOS_TOKEN,
    SPECTRUM_TOKEN_OFFSET,
    SpectrumTransformer,
)
from src.tokenizers.redshift import RedshiftTokenizer  # noqa: E402
from src.tokenizers.spectrum import SpectrumTokenizer  # noqa: E402
from src.training.utils import compute_metrics  # noqa: E402

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
    p.add_argument("--val-frac", type=float, default=0.02)
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

    # Logging
    p.add_argument("--run-name", type=str, default="approach_a")
    p.add_argument("--scratch-out", type=Path,
                   default=Path(os.environ.get("SCRATCH", "/tmp")) / "deepsrch")
    p.add_argument("--cfs-out", type=Path, default=None)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=2000)

    p.add_argument("--smoke", action="store_true",
                   help="Tiny config: 100 steps, 200 spectra, smaller model")
    return p.parse_args()


def lr_at(step: int, base_lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def tokenize_and_build(raw_batch, spec_tok, z_tok, approach, device):
    """Worker batch -> transformer-ready (encoder, decoder_in, target) on device.

    All sequences are fixed-length (the tokenizer interpolates to a fixed
    grid before encoding), so no padding masks are needed.
    """
    flux = raw_batch["flux"].to(device, non_blocking=True)
    ivar = raw_batch["ivar"].to(device, non_blocking=True)
    z_vals = raw_batch["z"]  # CPU (B,)

    istd = torch.sqrt(ivar.clamp(min=1e-10))
    x = torch.stack([flux, istd], dim=1)  # (B, 2, L)

    with torch.no_grad():
        spec_indices, _ = spec_tok.encode(x)  # (B, n_tokens)
    # Some tokenizer impls return (B, 1, n_tokens) -- squeeze if present.
    if spec_indices.dim() == 3:
        spec_indices = spec_indices.squeeze(1)
    spec_tokens = spec_indices.long() + SPECTRUM_TOKEN_OFFSET

    redshift_idx = torch.tensor(
        [z_tok.encode(float(z)) for z in z_vals.tolist()],
        dtype=torch.long, device=device,
    )
    redshift_tokens = redshift_idx + REDSHIFT_TOKEN_OFFSET

    B = flux.shape[0]
    sos = torch.full((B, 1), SOS_TOKEN, dtype=torch.long, device=device)
    eos = torch.full((B, 1), EOS_TOKEN, dtype=torch.long, device=device)
    rz = redshift_tokens.unsqueeze(1)  # (B, 1)

    if approach == "a":
        encoder_input = torch.cat([sos, rz, spec_tokens, eos], dim=1)
    else:  # 'b'
        encoder_input = torch.cat([sos, spec_tokens, eos], dim=1)

    decoder_input = torch.cat([sos, rz, spec_tokens], dim=1)
    target = torch.cat([rz, spec_tokens, eos], dim=1)
    return encoder_input, decoder_input, target


@torch.no_grad()
def evaluate(model, loader, spec_tok, z_tok, approach, device, amp, max_batches=50):
    model.eval()
    losses = 0.0
    metrics_accum = {"overall_acc": 0.0, "redshift_acc": 0.0, "spectrum_acc": 0.0}
    n = 0
    for i, raw in enumerate(loader):
        if raw is None:
            continue
        if i >= max_batches:
            break
        enc, dec, tgt = tokenize_and_build(raw, spec_tok, z_tok, approach, device)
        with torch.amp.autocast("cuda", enabled=amp):
            logits, loss = model(enc, dec, targets=tgt)
        losses += float(loss.item())
        m = compute_metrics(logits, tgt)
        for k in metrics_accum:
            metrics_accum[k] += m[k]
        n += 1
    if n == 0:
        return {"loss": float("nan"), **{k: float("nan") for k in metrics_accum}}
    return {"loss": losses / n, **{k: v / n for k, v in metrics_accum.items()}}


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device} approach={args.approach} steps={args.steps}")
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

    # Pretrained spectrum tokenizer (lives on GPU in main process; never forked)
    print(f"[tok] loading spectrum tokenizer {args.tokenizer_ckpt}")
    spec_tok = SpectrumTokenizer().to(device)
    ckpt = torch.load(args.tokenizer_ckpt, map_location=device, weights_only=False)
    sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    spec_tok.load_state_dict(sd)
    spec_tok.eval()
    for p in spec_tok.parameters():
        p.requires_grad_(False)

    # Fit redshift tokenizer on a sample of manifest redshifts
    print(f"[tok] fitting redshift tokenizer on up to {args.z_fit_files} redrock files")
    zs = collect_redshifts(records, max_files=args.z_fit_files)
    print(f"[tok]   gathered {len(zs)} z values, min={zs.min():.4f} max={zs.max():.4f}")
    z_tok = RedshiftTokenizer(n_levels=256)
    z_tok.fit(zs)

    # Datasets -- raw spectra only (CPU). Workers safe to fork.
    base = DR1IndexedDataset(
        records,
        require_good_zwarn=True,
        require_nonzero_flux=True,
        max_spectra=args.max_spectra,
    )
    print(f"[data] {len(base)} spectra in flat index")

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(base), generator=g).tolist()
    n_val = max(1, int(len(base) * args.val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_ds, val_ds = Subset(base, train_idx), Subset(base, val_idx)
    print(f"[data] train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
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
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params:,} (~{n_params/1e6:.1f}M)")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    step = 0
    best_val = float("inf")
    t0 = time.time()
    train_iter = iter(train_loader)
    model.train()

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

        enc, dec, tgt = tokenize_and_build(raw, spec_tok, z_tok, args.approach, device)

        optim.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=args.amp):
            logits, loss = model(enc, dec, targets=tgt)
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optim)
        scaler.update()

        if step % args.log_every == 0:
            with torch.no_grad():
                m = compute_metrics(logits, tgt)
            dt = time.time() - t0
            rate = (step + 1) / max(dt, 1e-6)
            msg = {
                "kind": "train", "step": step, "lr": optim.param_groups[0]["lr"],
                "loss": float(loss.item()),
                **m, "steps_per_sec": rate, "elapsed_s": dt,
            }
            print(f"[step {step:6d}] loss={msg['loss']:.4f} "
                  f"acc={m['overall_acc']:.3f} z_acc={m['redshift_acc']:.3f} "
                  f"spec_acc={m['spectrum_acc']:.3f} {rate:.1f} step/s")
            with metrics_path.open("a") as f:
                f.write(json.dumps(msg) + "\n")

        if step > 0 and step % args.val_every == 0:
            v = evaluate(model, val_loader, spec_tok, z_tok, args.approach, device, args.amp)
            model.train()
            print(f"[val   {step:6d}] " + " ".join(f"{k}={v[k]:.4f}" for k in v))
            with metrics_path.open("a") as f:
                f.write(json.dumps({"kind": "val", "step": step,
                                    **{f"val_{k}": vv for k, vv in v.items()}}) + "\n")
            if v["loss"] < best_val:
                best_val = v["loss"]
                p = run_dir / "best.pt"
                torch.save({
                    "step": step,
                    "model": model.state_dict(),
                    "optim": optim.state_dict(),
                    "scaler": scaler.state_dict(),
                    "val_loss": best_val,
                }, p)
                print(f"  *** new best val_loss={best_val:.4f} -> {p}")
                if args.cfs_out is not None:
                    args.cfs_out.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, args.cfs_out / "best.pt")

        if step > 0 and step % args.save_every == 0:
            p = run_dir / f"step_{step:08d}.pt"
            torch.save({"step": step, "model": model.state_dict()}, p)
            print(f"  ckpt -> {p}")

        step += 1

    p = run_dir / "final.pt"
    torch.save({"step": step, "model": model.state_dict()}, p)
    print(f"[done] final -> {p}  best_val_loss={best_val:.4f}")
    if args.cfs_out is not None:
        args.cfs_out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, args.cfs_out / "final.pt")


if __name__ == "__main__":
    main()
