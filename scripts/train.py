"""
Train Spectrum Transformer
===========================
Training script for Approach A (joint) or Approach B (masked).

Usage:
    # Approach A (joint redshift prediction)
    python scripts/train.py --approach a --epochs 10 --batch_size 4

    # Approach B (masked redshift)
    python scripts/train.py --approach b --epochs 10 --batch_size 4

    # Resume from checkpoint
    python scripts/train.py --approach a --resume checkpoints/approach_a_latest.pt
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.transformer import SpectrumTransformer
from src.tokenizers.spectrum import SpectrumTokenizer
from src.tokenizers.redshift import RedshiftTokenizer
from src.datasets.tokenized_dataset import TokenizedSpectrumDataset, collate_tokenized_batch
from src.training.utils import (
    AverageMeter,
    compute_metrics,
    save_checkpoint,
    load_checkpoint,
    log_metrics,
)
from src.utils.data import DESISpectrumDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train spectrum transformer')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='data/desi_raw',
                        help='Directory with coadd and redrock FITS files')
    parser.add_argument('--approach', type=str, choices=['a', 'b'], default='a',
                        help='Training approach: a=joint, b=masked')
    
    # Model
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--n_encoder_layers', type=int, default=6)
    parser.add_argument('--n_decoder_layers', type=int, default=6)
    parser.add_argument('--n_heads', type=int, default=12)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='auto',
                        help='auto, cpu, cuda, or mps')
    
    # Logging
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--log_every', type=int, default=10,
                        help='Log metrics every N batches')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Smoke test
    parser.add_argument('--smoke_test', action='store_true',
                        help='Quick 3-epoch test on small data')
    
    return parser.parse_args()


def get_device(device_str: str):
    """Get torch device."""
    if device_str == 'auto':
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    return torch.device(device_str)


def load_data(data_dir: Path, require_good_zwarn: bool = True):
    """Load all spectra from coadd files."""
    coadd_files = sorted(data_dir.glob('coadd-*.fits'))
    
    all_spectra = []
    for f in coadd_files:
        rr = f.parent / f.name.replace('coadd-', 'redrock-')
        ds = DESISpectrumDataset(
            coadd_path=f,
            redrock_path=rr,
            require_good_zwarn=require_good_zwarn,
            require_nonzero_flux=True,
        )
        for i in range(len(ds)):
            all_spectra.append(ds[i])
    
    return all_spectra


def train_epoch(model, dataloader, optimizer, device, grad_clip, log_every):
    """Train for one epoch."""
    model.train()
    
    loss_meter = AverageMeter()
    metrics_accum = {'overall_acc': 0, 'redshift_acc': 0, 'spectrum_acc': 0}
    
    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, batch in enumerate(pbar):
        encoder_input = batch['encoder_input'].to(device)
        decoder_input = batch['decoder_input'].to(device)
        target = batch['target'].to(device)
        encoder_mask = batch['encoder_mask'].to(device)
        decoder_mask = batch['decoder_mask'].to(device)
        
        optimizer.zero_grad()
        
        logits, loss = model(
            encoder_input,
            decoder_input,
            encoder_mask=encoder_mask,
            decoder_mask=decoder_mask,
            targets=target,
        )
        
        loss.backward()
        
        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        loss_meter.update(loss.item(), encoder_input.size(0))
        
        # Compute metrics
        with torch.no_grad():
            metrics = compute_metrics(logits, target)
            for k in metrics_accum:
                metrics_accum[k] += metrics[k]
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc': f'{metrics["overall_acc"]:.3f}',
        })
    
    # Average metrics
    n_batches = len(dataloader)
    for k in metrics_accum:
        metrics_accum[k] /= n_batches
    
    return loss_meter.avg, metrics_accum


@torch.no_grad()
def validate(model, dataloader, device):
    """Validate for one epoch."""
    model.eval()
    
    loss_meter = AverageMeter()
    metrics_accum = {'overall_acc': 0, 'redshift_acc': 0, 'spectrum_acc': 0}
    
    for batch in tqdm(dataloader, desc='Validation'):
        encoder_input = batch['encoder_input'].to(device)
        decoder_input = batch['decoder_input'].to(device)
        target = batch['target'].to(device)
        encoder_mask = batch['encoder_mask'].to(device)
        decoder_mask = batch['decoder_mask'].to(device)
        
        logits, loss = model(
            encoder_input,
            decoder_input,
            encoder_mask=encoder_mask,
            decoder_mask=decoder_mask,
            targets=target,
        )
        
        loss_meter.update(loss.item(), encoder_input.size(0))
        
        metrics = compute_metrics(logits, target)
        for k in metrics_accum:
            metrics_accum[k] += metrics[k]
    
    n_batches = len(dataloader)
    for k in metrics_accum:
        metrics_accum[k] /= n_batches
    
    return loss_meter.avg, metrics_accum


def main():
    args = parse_args()
    
    # Setup
    device = get_device(args.device)
    print(f"Device: {device}")
    
    save_dir = Path(args.save_dir) / f'approach_{args.approach}'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load data
    print(f"\nLoading data from {args.data_dir}...")
    data_dir = Path(args.data_dir)
    all_spectra = load_data(data_dir, require_good_zwarn=False)
    print(f"Loaded {len(all_spectra)} spectra")
    
    if args.smoke_test:
        print("SMOKE TEST: Using only 50 spectra, 3 epochs")
        all_spectra = all_spectra[:50]
        args.epochs = 3
    
    # Split train/val (90/10) with fixed random seed for reproducibility
    import random
    random.seed(42)
    indices = list(range(len(all_spectra)))
    random.shuffle(indices)
    n_train = int(0.9 * len(all_spectra))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    train_spectra = [all_spectra[i] for i in train_idx]
    val_spectra = [all_spectra[i] for i in val_idx]
    print(f"Train: {len(train_spectra)}, Val: {len(val_spectra)}")
    
    # Tokenizers
    print("\nInitializing tokenizers...")
    spectrum_tokenizer = SpectrumTokenizer().to(device)
    
    # Fit redshift tokenizer
    all_z = [s['z'] for s in all_spectra]
    redshift_tokenizer = RedshiftTokenizer(n_levels=256)
    redshift_tokenizer.fit(all_z)
    print(f"Redshift tokenizer fitted on {len(all_z)} samples")
    print(f"  Range: [{redshift_tokenizer._min_z:.4f}, {redshift_tokenizer._max_z:.4f}]")
    
    # Datasets
    train_dataset = TokenizedSpectrumDataset(
        train_spectra,
        spectrum_tokenizer,
        redshift_tokenizer,
        approach=args.approach,
        device=device,
    )
    val_dataset = TokenizedSpectrumDataset(
        val_spectra,
        spectrum_tokenizer,
        redshift_tokenizer,
        approach=args.approach,
        device=device,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_tokenized_batch,
        num_workers=0,  # Must be 0 for MPS
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_tokenized_batch,
        num_workers=0,
    )
    
    # Model
    print(f"\nInitializing model...")
    model = SpectrumTransformer(
        d_model=args.d_model,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters (~{n_params/1e6:.1f}M)")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Resume
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, Path(args.resume), device)
        start_epoch += 1
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Training Approach {args.approach.upper()}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"{'='*60}\n")
    
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            args.grad_clip, args.log_every,
        )
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, device)
        
        epoch_time = time.time() - epoch_start
        
        # Print summary
        print(f"\nEpoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train: loss={train_loss:.4f}, acc={train_metrics['overall_acc']:.3f}, "
              f"redshift_acc={train_metrics['redshift_acc']:.3f}")
        print(f"  Val:   loss={val_loss:.4f}, acc={val_metrics['overall_acc']:.3f}, "
              f"redshift_acc={val_metrics['redshift_acc']:.3f}")
        
        # Log metrics
        log_metrics({
            'train_loss': train_loss,
            **{f'train_{k}': v for k, v in train_metrics.items()},
            'val_loss': val_loss,
            **{f'val_{k}': v for k, v in val_metrics.items()},
        }, save_dir / 'metrics.json', epoch + 1)
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            save_checkpoint(
                model, optimizer, epoch + 1, val_loss,
                save_dir, prefix=f'approach_{args.approach}',
            )
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch + 1, val_loss,
                save_dir, prefix=f'approach_{args.approach}_best',
            )
            print(f"  *** New best validation loss: {val_loss:.4f} ***")
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {save_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
