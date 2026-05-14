"""
Training utilities for spectrum transformer.
"""

import os
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
from pathlib import Path

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None


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


def compute_masked_redshift_acc(
    logits: torch.Tensor,
    target: torch.Tensor,
    rz_mask: torch.Tensor,
) -> Dict[str, float]:
    """Redshift accuracy restricted to samples where the encoder's rz was masked.

    This is the honest redshift metric — accuracy at decoder position 0 only
    for samples where the encoder received [MASK] instead of the actual
    redshift token. Computed only over positions where the model could not
    trivially copy from the encoder.

    Args:
        logits: (B, L, V)
        target: (B, L) with -100 for ignored positions
        rz_mask: (B, 1) bool — True where encoder rz was replaced with MASK

    Returns:
        {'redshift_acc_masked': float, 'n_rz_masked': int}
        redshift_acc_masked is NaN if no samples were masked in the batch.
    """
    pred = logits.argmax(dim=-1)  # (B, L)
    valid = (target[:, 0] != -100) & rz_mask.squeeze(-1)
    n = int(valid.sum().item())
    if n == 0:
        return {'redshift_acc_masked': float('nan'), 'n_rz_masked': 0}
    correct = ((pred[:, 0] == target[:, 0]) & valid).sum().float()
    return {
        'redshift_acc_masked': (correct / n).item(),
        'n_rz_masked': n,
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


# ---------------------------------------------------------------------------
# AUC and R² metrics for AION benchmarking
# ---------------------------------------------------------------------------

def _squeeze_to_1d(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1)


def compute_masked_auc(
    logits: torch.Tensor,
    target: torch.Tensor,
    masked_positions: torch.Tensor,
    n_negatives: int = 10,
) -> Dict[str, float]:
    """AUC at encoder-masked positions.

    For each masked spectrum position, treats prediction as binary:
      - Positive: softmax probability assigned to the correct token
      - Negative: mean softmax probability of K random wrong tokens
    Per-position AUC = P(positive > negative), then average across positions.

    A value of 1.0 means the correct token had higher probability than all
    sampled wrong tokens at every masked position; 0.0 means the model never
    ranked the correct token above a random wrong token.

    Args:
        logits: (B, L, V) raw unnormalized scores
        target: (B, L) token IDs, -100 for ignored positions
        masked_positions: (B, T_spec) bool, True where encoder was masked
        n_negatives: K random wrong tokens sampled per position

    Returns:
        {'mean_mask_auc': float, 'n_masked': int}
        mean_mask_auc is NaN if no positions were masked or sklearn unavailable.
    """
    if roc_auc_score is None:
        return {'mean_mask_auc': float('nan'), 'n_masked': 0}

    probs = torch.softmax(logits.float(), dim=-1)  # (B, L, V)
    B, T_spec = masked_positions.shape
    spec_target = target[:, 1:1 + T_spec]  # (B, T_spec)
    spec_probs = probs[:, 1:1 + T_spec, :]  # (B, T_spec, V)
    valid = (spec_target != -100) & masked_positions.bool()
    n_masked = int(valid.sum().item())
    if n_masked == 0:
        return {'mean_mask_auc': float('nan'), 'n_masked': 0}

    # Collect per-position (y_true, y_score) for sklearn's binary AUC.
    # y_true=1 for correct token, y_true=0 for each sampled negative.
    y_true_list = []
    y_score_list = []
    V = logits.shape[-1]

    for b in range(B):
        for j in range(T_spec):
            if not valid[b, j]:
                continue
            correct_tok = int(spec_target[b, j].item())
            p_correct = float(spec_probs[b, j, correct_tok].item())
            # Sample K unique wrong tokens
            all_toks = list(range(V))
            try:
                wrong_toks = random.sample(
                    [t for t in all_toks if t != correct_tok],
                    k=min(n_negatives, V - 1),
                )
            except ValueError:
                wrong_toks = []
            y_true_list.append(1)
            y_score_list.append(p_correct)
            for _ in wrong_toks:
                y_true_list.append(0)
                y_score_list.append(float(spec_probs[b, j, _].item()))

    mean_auc = float(roc_auc_score(y_true_list, y_score_list))
    return {'mean_mask_auc': mean_auc, 'n_masked': n_masked}


def compute_masked_r2(
    logits: torch.Tensor,
    target: torch.Tensor,
    masked_positions: torch.Tensor,
) -> Dict[str, float]:
    """R² at encoder-masked positions.

    Uses the binary R² formula:
        R² = 1 - BCE / BCE_null
    where BCE = mean(-log(p_correct)) over masked positions and
    BCE_null = -log(1/T) with T=vocab_size (the entropy of uniform).

    Interpretation:
        1.0  = perfect prediction (p_correct → 1)
        0.0  = same as random guessing (p_correct = 1/T)
      < 0.0  = worse than random (model actively wrong)

    Args:
        logits: (B, L, V)
        target: (B, L) with -100 for ignored
        masked_positions: (B, T_spec) bool, True where encoder was masked

    Returns:
        {'masked_spec_r2': float, 'n_masked': int}
        masked_spec_r2 is NaN if no positions were masked.
    """
    probs = torch.softmax(logits.float(), dim=-1)  # (B, L, V)
    B, T_spec = masked_positions.shape
    spec_target = target[:, 1:1 + T_spec]
    correct_probs = probs[:, 1:1 + T_spec, :].gather(
        dim=-1, index=spec_target.unsqueeze(-1),
    ).squeeze(-1)  # (B, T_spec)
    valid = (spec_target != -100) & masked_positions.bool()
    n_masked = int(valid.sum().item())
    if n_masked == 0:
        return {'masked_spec_r2': float('nan'), 'n_masked': 0}

    p_correct = correct_probs[valid].clamp(min=1e-10)
    T = float(logits.shape[-1])
    bce = -(p_correct.log()).mean()
    bce_null = -(torch.tensor(1.0 / T, device=logits.device).log())
    r2 = 1.0 - (bce / bce_null).clamp(max=1.0)
    return {'masked_spec_r2': float(r2.item()), 'n_masked': n_masked}


def compute_all_auc(
    logits: torch.Tensor,
    target: torch.Tensor,
    n_negatives: int = 10,
) -> Dict[str, float]:
    """AUC at all valid spectrum positions (no masking requirement).

    Same methodology as compute_masked_auc but computed over every position
    where target[:, 1:] is not -100. Useful for overall quality assessment.

    Args:
        logits: (B, L, V)
        target: (B, L) with -100 for ignored positions
        n_negatives: K random wrong tokens sampled per position

    Returns:
        {'all_mean_auc': float, 'n_positions': int}
        all_mean_auc is NaN if no positions or sklearn unavailable.
    """
    if roc_auc_score is None:
        return {'all_mean_auc': float('nan'), 'n_positions': 0}

    probs = torch.softmax(logits.float(), dim=-1)
    B, L = target.shape
    # Spectrum positions are target indices 1..L-1 (excluding position 0=rz and last=EOS)
    spec_target = target[:, 1:]  # (B, L-1)
    spec_probs = probs[:, 1:, :]  # (B, L-1, V)
    valid = spec_target != -100  # (B, L-1)
    n_positions = int(valid.sum().item())
    if n_positions == 0:
        return {'all_mean_auc': float('nan'), 'n_positions': 0}

    y_true_list, y_score_list = [], []
    V = logits.shape[-1]
    for b in range(B):
        for j in range(L - 1):
            if not valid[b, j]:
                continue
            correct_tok = int(spec_target[b, j].item())
            p_correct = float(spec_probs[b, j, correct_tok].item())
            try:
                wrong_toks = random.sample(
                    [t for t in range(V) if t != correct_tok],
                    k=min(n_negatives, V - 1),
                )
            except ValueError:
                wrong_toks = []
            y_true_list.append(1)
            y_score_list.append(p_correct)
            for _ in wrong_toks:
                y_true_list.append(0)
                y_score_list.append(float(spec_probs[b, j, _].item()))

    mean_auc = float(roc_auc_score(y_true_list, y_score_list))
    return {'all_mean_auc': mean_auc, 'n_positions': n_positions}


def compute_all_r2(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    """R² at all valid spectrum positions (no masking requirement).

    Same methodology as compute_masked_r2 but computed over every position
    where target[:, 1:] is not -100.

    Args:
        logits: (B, L, V)
        target: (B, L) with -100 for ignored positions

    Returns:
        {'all_spec_r2': float, 'n_positions': int}
    """
    probs = torch.softmax(logits.float(), dim=-1)
    B, L = target.shape
    spec_target = target[:, 1:]  # (B, L-1)
    valid = spec_target != -100  # (B, L-1)
    n_positions = int(valid.sum().item())
    if n_positions == 0:
        return {'all_spec_r2': float('nan'), 'n_positions': 0}
    # Gather only at valid positions
    valid_target = spec_target.clone()
    valid_target[~valid] = 0  # temporary fill for gather; ignored via valid mask
    correct_probs = probs[:, 1:, :].gather(
        dim=-1, index=valid_target.unsqueeze(-1),
    ).squeeze(-1)  # (B, L-1)
    p_correct = correct_probs[valid].clamp(min=1e-10)
    T = float(logits.shape[-1])
    bce = -(p_correct.log()).mean()
    bce_null = -(torch.tensor(1.0 / T, device=logits.device).log())
    r2 = 1.0 - (bce / bce_null).clamp(max=1.0)
    return {'all_spec_r2': float(r2.item()), 'n_positions': n_positions}
