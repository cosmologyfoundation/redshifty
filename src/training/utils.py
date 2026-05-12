"""
Training utilities for spectrum transformer.
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
from pathlib import Path


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_metrics(logits: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Compute training metrics.
    
    Args:
        logits: (B, L, vocab_size)
        target: (B, L) with -100 for ignored positions
        
    Returns:
        Dict with accuracy metrics
    """
    # Get predictions
    pred = logits.argmax(dim=-1)  # (B, L)
    
    # Mask out ignored positions
    mask = (target != -100)
    
    # Overall accuracy
    correct = (pred == target).float() * mask.float()
    total = mask.float().sum()
    overall_acc = correct.sum() / total if total > 0 else 0.0
    
    # Position 0 accuracy (redshift token)
    pos0_mask = mask[:, 0]
    pos0_correct = (pred[:, 0] == target[:, 0]).float() * pos0_mask.float()
    pos0_acc = pos0_correct.sum() / pos0_mask.sum() if pos0_mask.sum() > 0 else 0.0
    
    # Position 1+ accuracy (spectrum tokens)
    if target.shape[1] > 1:
        spec_mask = mask[:, 1:]
        spec_correct = (pred[:, 1:] == target[:, 1:]).float() * spec_mask.float()
        spec_acc = spec_correct.sum() / spec_mask.sum() if spec_mask.sum() > 0 else 0.0
    else:
        spec_acc = 0.0
    
    return {
        'overall_acc': overall_acc.item(),
        'redshift_acc': pos0_acc.item(),
        'spectrum_acc': spec_acc.item(),
    }


def compute_masked_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    masked_positions: torch.Tensor,
) -> Dict[str, float]:
    """Spectrum accuracy restricted to encoder-masked positions.

    This is the honest spectrum-reconstruction metric — accuracy at the
    decoder positions whose corresponding encoder position was replaced
    with `MASK_TOKEN`. Computed only over positions where the model
    could not trivially copy from the encoder.

    Layout assumption (matches `tokenize_and_build`):
      target = [redshift, s_1, s_2, ..., s_T, EOS]                 (B, T+2)
      masked_positions[i, j] is True iff encoder pos for `s_{j+1}`
        was masked. So target index for masked_positions[i, j] is j+1.

    Args:
        logits: (B, L, V)
        target: (B, L) with -100 for ignored positions
        masked_positions: (B, T) bool

    Returns:
        {'masked_spec_acc': float, 'n_masked': int}
        masked_spec_acc is NaN if no positions were masked in the batch.
    """
    pred = logits.argmax(dim=-1)  # (B, L)
    B, T_spec = masked_positions.shape
    # Spectrum-token target positions live at indices [1, 1+T_spec).
    spec_target = target[:, 1:1 + T_spec]
    spec_pred = pred[:, 1:1 + T_spec]
    valid = (spec_target != -100) & masked_positions.bool()
    n_masked = int(valid.sum().item())
    if n_masked == 0:
        return {'masked_spec_acc': float('nan'), 'n_masked': 0}
    correct = ((spec_pred == spec_target) & valid).sum().float()
    return {
        'masked_spec_acc': (correct / n_masked).item(),
        'n_masked': n_masked,
    }


def compute_loss_breakdown(logits: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Unweighted per-segment loss for logging.

    Partitions the cross-entropy loss into the position-0 (redshift) and
    position-1+ (spectrum) contributions, each reduced to a mean over
    valid (non -100) positions. Returns three floats; useful for tracking
    whether the redshift branch is actually learning even when the
    backprop loss uses a non-unit redshift_weight.
    """
    B, L = target.shape
    per_token = F.cross_entropy(
        logits.view(-1, logits.shape[-1]),
        target.view(-1),
        ignore_index=-100,
        reduction='none',
    ).view(B, L)
    valid = (target != -100).float()
    n_red = valid[:, 0].sum().clamp(min=1.0)
    loss_red = (per_token[:, 0] * valid[:, 0]).sum() / n_red
    if L > 1:
        n_spec = valid[:, 1:].sum().clamp(min=1.0)
        loss_spec = (per_token[:, 1:] * valid[:, 1:]).sum() / n_spec
    else:
        loss_spec = torch.zeros((), device=logits.device)
    # "Total" here is the unweighted mean over all valid positions; this is
    # the comparable-to-vanilla number, not what backprop uses.
    n_all = valid.sum().clamp(min=1.0)
    loss_total = (per_token * valid).sum() / n_all
    return {
        'loss_redshift': loss_red.item(),
        'loss_spectrum': loss_spec.item(),
        'loss_total': loss_total.item(),
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    save_dir: Path,
    prefix: str = 'checkpoint',
):
    """Save training checkpoint."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    path = save_dir / f'{prefix}_epoch{epoch:04d}.pt'
    torch.save(checkpoint, path)
    print(f"  Checkpoint saved: {path}")
    
    # Also save latest
    latest_path = save_dir / f'{prefix}_latest.pt'
    torch.save(checkpoint, latest_path)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_path: Path,
    device: torch.device,
) -> int:
    """Load training checkpoint. Returns starting epoch."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
    print(f"Loaded checkpoint from epoch {epoch} (loss={loss:.4f})")
    return epoch


def log_metrics(
    metrics: Dict[str, float],
    log_file: Path,
    epoch: int,
    split: str = 'train',
):
    """Append metrics to JSON log file."""
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    entry = {
        'epoch': epoch,
        'split': split,
        'timestamp': time.time(),
        **metrics,
    }
    
    # Read existing or create new
    if log_file.exists():
        with open(log_file, 'r') as f:
            logs = json.load(f)
    else:
        logs = []
    
    logs.append(entry)
    
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=2)
