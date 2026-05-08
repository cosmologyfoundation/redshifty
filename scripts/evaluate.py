"""
Evaluate Redshift Prediction
==============================
Load a trained model and evaluate redshift prediction accuracy.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/approach_a_20ep/approach_a/approach_a_best_epoch0015.pt --approach a
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import torch
import numpy as np
from torch.utils.data import DataLoader

from src.models.transformer import (
    SpectrumTransformer,
    REDSHIFT_TOKEN_OFFSET,
)
from src.tokenizers.spectrum import SpectrumTokenizer
from src.tokenizers.redshift import RedshiftTokenizer
from src.datasets.tokenized_dataset import TokenizedSpectrumDataset, collate_tokenized_batch
from src.utils.data import DESISpectrumDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate redshift prediction')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/desi_raw')
    parser.add_argument('--approach', type=str, choices=['a', 'b'], required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--device', type=str, default='auto')
    return parser.parse_args()


def get_device(device_str):
    if device_str == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
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


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Load config from config.json
    config_path = Path(args.checkpoint).parent / 'config.json'
    if config_path.exists():
        config = json.load(open(config_path))
    else:
        config = {
            'data_dir': args.data_dir,
            'd_model': 256,
            'n_encoder_layers': 2,
            'n_decoder_layers': 2,
            'n_heads': 8,
            'dropout': 0.1,
        }
    
    # Load data
    print(f"Loading data from {config['data_dir']}...")
    all_spectra = load_data(Path(config['data_dir']), require_good_zwarn=False)
    print(f"Loaded {len(all_spectra)} spectra")
    
    # Train/val split (same as training: 90/10, seed=42)
    n = len(all_spectra)
    n_train = int(0.9 * n)
    torch.manual_seed(42)
    perm = torch.randperm(n)
    train_idx = perm[:n_train].tolist()
    val_idx = perm[n_train:].tolist()
    val_spectra = [all_spectra[i] for i in val_idx]
    print(f"Val: {len(val_spectra)} spectra")
    
    # Tokenizers
    spectrum_tok = SpectrumTokenizer().to(device)
    redshift_tok = RedshiftTokenizer(n_levels=256)
    redshift_tok.fit([s['z'] for s in all_spectra])
    
    # Dataset
    val_dataset = TokenizedSpectrumDataset(
        val_spectra,
        spectrum_tok,
        redshift_tok,
        approach=args.approach,
        device=device,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_tokenized_batch,
        num_workers=0,
    )
    
    # Build model
    model = SpectrumTransformer(
        d_model=config['d_model'],
        n_encoder_layers=config['n_encoder_layers'],
        n_decoder_layers=config['n_decoder_layers'],
        n_heads=config.get('n_heads', 12),
        dropout=config.get('dropout', 0.1),
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Evaluate
    print("\nEvaluating...")
    all_true_z = []
    all_pred_z = []
    all_true_tokens = []
    all_pred_tokens = []
    
    with torch.no_grad():
        for batch in val_loader:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            target = batch['target'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            
            logits, _ = model(encoder_input, decoder_input, encoder_mask, decoder_mask)
            pred_tokens = logits.argmax(dim=-1)
            
            batch_size = encoder_input.shape[0]
            for b in range(batch_size):
                true_token = target[b, 0].item()
                pred_token = pred_tokens[b, 0].item()
                
                true_z = redshift_tok.decode([true_token - REDSHIFT_TOKEN_OFFSET])[0]
                pred_z = redshift_tok.decode([pred_token - REDSHIFT_TOKEN_OFFSET])[0]
                
                all_true_z.append(true_z)
                all_pred_z.append(pred_z)
                all_true_tokens.append(true_token)
                all_pred_tokens.append(pred_token)
    
    all_true_z = np.array(all_true_z)
    all_pred_z = np.array(all_pred_z)
    all_true_tokens = np.array(all_true_tokens)
    all_pred_tokens = np.array(all_pred_tokens)
    
    token_acc = (all_true_tokens == all_pred_tokens).mean()
    mae = np.abs(all_true_z - all_pred_z).mean()
    rmse = np.sqrt(((all_true_z - all_pred_z) ** 2).mean())
    
    print("\n" + "="*60)
    print("REDSHIFT PREDICTION RESULTS")
    print("="*60)
    print(f"Token Accuracy:      {token_acc*100:.2f}%")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"RMSE:                {rmse:.4f}")
    print(f"Correct / Total:     {(all_true_tokens == all_pred_tokens).sum()} / {len(all_true_tokens)}")
    print("="*60)
    
    print("\nIndividual Predictions:")
    print(f"{'Index':>6} {'True z':>10} {'Pred z':>10} {'Error':>10} {'Match':>6}")
    print("-" * 50)
    for i in range(len(all_true_z)):
        true = all_true_z[i]
        pred = all_pred_z[i]
        error = abs(true - pred)
        match = "YES" if all_true_tokens[i] == all_pred_tokens[i] else "NO"
        print(f"{i:6d} {true:10.4f} {pred:10.4f} {error:10.4f} {match:>6}")


if __name__ == '__main__':
    main()
