"""
Autoregressive Generation Test
==============================
Load a trained model and generate spectra token-by-token without teacher forcing.

Usage:
    python scripts/generate.py --checkpoint checkpoints/approach_b_fixed_split/approach_b/approach_b_best_epoch0010.pt --approach b --n_samples 5
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.models.transformer import (
    SpectrumTransformer,
    REDSHIFT_TOKEN_OFFSET,
    SPECTRUM_TOKEN_OFFSET,
    EOS_TOKEN,
    SOS_TOKEN,
)
from src.tokenizers.spectrum import SpectrumTokenizer
from src.tokenizers.redshift import RedshiftTokenizer
from src.datasets.tokenized_dataset import TokenizedSpectrumDataset, collate_tokenized_batch
from src.utils.data import DESISpectrumDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Autoregressive generation test')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='data/desi_raw')
    parser.add_argument('--approach', type=str, choices=['a', 'b'], required=True)
    parser.add_argument('--n_samples', type=int, default=5, help='Number of spectra to generate')
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


def decode_redshift_token(token_id, redshift_tok):
    """Decode a redshift token ID back to a scalar z value."""
    fsq_index = token_id - REDSHIFT_TOKEN_OFFSET
    if fsq_index < 0 or fsq_index >= redshift_tok.n_levels:
        return None
    return redshift_tok.decode(fsq_index)


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    config_path = Path(args.checkpoint).parent / 'config.json'
    if config_path.exists():
        config = json.load(open(config_path))
    else:
        config = {
            'd_model': 256,
            'n_encoder_layers': 2,
            'n_decoder_layers': 2,
            'n_heads': 8,
            'dropout': 0.1,
        }
    
    # Load data
    print(f"Loading data from {args.data_dir}...")
    all_spectra = load_data(Path(args.data_dir), require_good_zwarn=False)
    print(f"Loaded {len(all_spectra)} spectra")
    
    # Train/val split (same as training: random 90/10, seed=42)
    random.seed(42)
    indices = list(range(len(all_spectra)))
    random.shuffle(indices)
    n_train = int(0.9 * len(all_spectra))
    val_idx = indices[n_train:]
    val_spectra = [all_spectra[i] for i in val_idx]
    print(f"Val: {len(val_spectra)} spectra")
    
    # Tokenizers
    spectrum_tok = SpectrumTokenizer().to(device)
    redshift_tok = RedshiftTokenizer(n_levels=256)
    redshift_tok.fit([s['z'] for s in all_spectra])
    
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
    print("Model loaded.")
    
    # Pick n_samples from validation
    random.seed(123)  # Different seed for sample selection
    sample_indices = random.sample(range(len(val_spectra)), min(args.n_samples, len(val_spectra)))
    
    print(f"\n{'='*60}")
    print(f"AUTOREGRESSIVE GENERATION ({args.n_samples} samples)")
    print(f"{'='*60}\n")
    
    for idx in sample_indices:
        spec = val_spectra[idx]
        true_z = spec['z']
        
        # Create dataset for this single spectrum to get encoder input
        single_ds = TokenizedSpectrumDataset(
            [spec],
            spectrum_tok,
            redshift_tok,
            approach=args.approach,
            device=device,
        )
        single_loader = DataLoader(
            single_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_tokenized_batch,
            num_workers=0,
        )
        batch = next(iter(single_loader))
        
        encoder_input = batch['encoder_input'].to(device)
        decoder_input = batch['decoder_input'].to(device)
        target = batch['target'].to(device)
        encoder_mask = batch['encoder_mask'].to(device)
        
        # Teacher-forced prediction (for comparison)
        with torch.no_grad():
            logits_tf, _ = model(encoder_input, decoder_input, encoder_mask, 
                                 batch['decoder_mask'].to(device))
            pred_tf = logits_tf.argmax(dim=-1)[0]  # (L_dec,)
        
        # Autoregressive generation
        with torch.no_grad():
            generated = model.generate(
                encoder_input,
                decoder_start_token=SOS_TOKEN,
                max_new_tokens=target.shape[1],
                temperature=1.0,
            )[0]  # (1 + generated_len,)
        
        # Decode results
        true_redshift_tok = target[0, 0].item()
        tf_redshift_tok = pred_tf[0].item()
        gen_redshift_tok = generated[1].item() if len(generated) > 1 else generated[0].item()
        
        true_z_decoded = float(decode_redshift_token(true_redshift_tok, redshift_tok)) if decode_redshift_token(true_redshift_tok, redshift_tok) is not None else None
        tf_z = float(decode_redshift_token(tf_redshift_tok, redshift_tok)) if decode_redshift_token(tf_redshift_tok, redshift_tok) is not None else None
        gen_z = float(decode_redshift_token(gen_redshift_tok, redshift_tok)) if decode_redshift_token(gen_redshift_tok, redshift_tok) is not None else None
        
        # Compute token-level accuracy for spectrum tokens
        true_spec_tokens = target[0, 1:].cpu().numpy()
        tf_spec_tokens = pred_tf[1:].cpu().numpy()
        
        # Match lengths for generated
        gen_spec_tokens = generated[2:].cpu().numpy()  # Skip SOS and redshift
        min_len = min(len(true_spec_tokens), len(gen_spec_tokens))
        gen_acc = np.mean(true_spec_tokens[:min_len] == gen_spec_tokens[:min_len]) * 100
        
        print(f"Sample {idx} (true z={true_z:.4f}):")
        print(f"  Teacher-forced: z={tf_z:.4f}")
        print(f"  Generated:      z={gen_z:.4f}")
        print(f"  Spectrum token acc (generated vs true): {gen_acc:.1f}%")
        print(f"  Generated length: {len(generated)} tokens")
        print()
    
    print(f"{'='*60}")
    print("Note: At 269 spectra, spectrum generation is near-random.")
    print("Redshift should match if the model learned the prior.")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
