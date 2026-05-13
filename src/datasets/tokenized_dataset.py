"""
Tokenized Spectrum Dataset
==========================
PyTorch Dataset that tokenizes spectra and redshifts for transformer training.

Produces encoder-decoder sequence pairs for Approach A or B.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional

from src.tokenizers.spectrum import SpectrumTokenizer
from src.tokenizers.redshift import RedshiftTokenizer
from src.models.transformer import (
    encode_spectrum_token,
    encode_redshift_token,
    build_approach_a_sequences,
    build_approach_b_sequences,
    MASK_TOKEN,
    PAD_TOKEN,
)


class TokenizedSpectrumDataset(Dataset):
    """Dataset that produces tokenized sequence pairs for transformer training.
    
    Args:
        spectra: List of spectrum dicts from DESISpectrumDataset
        spectrum_tokenizer: Trained SpectrumTokenizer (or untrained for smoke tests)
        redshift_tokenizer: Fitted RedshiftTokenizer
        approach: 'a' for joint redshift, 'b' for masked redshift
        device: Device for tokenization
    """
    
    def __init__(
        self,
        spectra: List[Dict],
        spectrum_tokenizer: SpectrumTokenizer,
        redshift_tokenizer: RedshiftTokenizer,
        approach: str = 'a',
        device: torch.device = torch.device('cpu'),
        encoder_mask_ratio: float = 0.0,
    ):
        self.spectra = spectra
        self.spectrum_tokenizer = spectrum_tokenizer.to(device)
        self.redshift_tokenizer = redshift_tokenizer
        self.approach = approach.lower()
        self.device = device
        self.encoder_mask_ratio = encoder_mask_ratio
        
        assert self.approach in ('a', 'b'), f"approach must be 'a' or 'b', got {approach}"
    
    def __len__(self):
        return len(self.spectra)
    
    def __getitem__(self, idx):
        spec = self.spectra[idx]
        
        # Tokenize spectrum
        flux = spec['flux'].unsqueeze(0).to(self.device)
        ivar = spec['ivar'].unsqueeze(0).to(self.device)
        istd = torch.sqrt(ivar.clamp(min=1e-10))
        x = torch.stack([flux, istd], dim=1)
        
        self.spectrum_tokenizer.eval()
        with torch.no_grad():
            indices, denorm = self.spectrum_tokenizer.encode(x)
        
        # indices: (1, L) -> squeeze to (L,)
        spectrum_tokens = encode_spectrum_token(indices.squeeze(0)).cpu()
        
        # Tokenize redshift
        z = float(spec['z'])
        redshift_idx = self.redshift_tokenizer.encode(z)
        redshift_token = encode_redshift_token(redshift_idx).cpu()
        
        # Build sequences
        if self.approach == 'a':
            enc, dec_in, target = build_approach_a_sequences(redshift_token, spectrum_tokens)
        else:
            enc, dec_in, target = build_approach_b_sequences(redshift_token, spectrum_tokens)
        
        # Stochastically mask the redshift token in the encoder (approach A only).
        rz_masked = False
        if self.approach == 'a' and self.encoder_mask_ratio > 0.0:
            if torch.rand(1).item() < self.encoder_mask_ratio:
                enc[1] = MASK_TOKEN
                rz_masked = True
        
        return {
            'encoder_input': enc,
            'decoder_input': dec_in,
            'target': target,
            'redshift': torch.tensor(float(z), dtype=torch.float32),
            'denorm': denorm.cpu(),
            'rz_masked': torch.tensor(rz_masked),
        }


def collate_tokenized_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function with padding for variable-length sequences.
    
    Args:
        batch: List of dicts from TokenizedSpectrumDataset.__getitem__
        
    Returns:
        Batched tensors with padding masks
    """
    batch_size = len(batch)
    
    # Find max lengths
    max_enc_len = max(len(item['encoder_input']) for item in batch)
    max_dec_len = max(len(item['decoder_input']) for item in batch)
    max_tgt_len = max(len(item['target']) for item in batch)
    
    # Pad encoder inputs
    encoder_input = torch.full((batch_size, max_enc_len), PAD_TOKEN, dtype=torch.long)
    encoder_mask = torch.zeros(batch_size, max_enc_len, dtype=torch.bool)  # True = padding
    
    for i, item in enumerate(batch):
        l = len(item['encoder_input'])
        encoder_input[i, :l] = item['encoder_input']
        encoder_mask[i, l:] = True
    
    # Pad decoder inputs
    decoder_input = torch.full((batch_size, max_dec_len), PAD_TOKEN, dtype=torch.long)
    decoder_mask = torch.zeros(batch_size, max_dec_len, dtype=torch.bool)
    
    for i, item in enumerate(batch):
        l = len(item['decoder_input'])
        decoder_input[i, :l] = item['decoder_input']
        decoder_mask[i, l:] = True
    
    # Pad targets
    target = torch.full((batch_size, max_tgt_len), -100, dtype=torch.long)
    
    for i, item in enumerate(batch):
        l = len(item['target'])
        target[i, :l] = item['target']
    
    return {
        'encoder_input': encoder_input,
        'decoder_input': decoder_input,
        'target': target,
        'encoder_mask': encoder_mask,
        'decoder_mask': decoder_mask,
        'redshift': torch.stack([item['redshift'] for item in batch]),
        'denorm': torch.stack([item['denorm'] for item in batch]),
        'rz_masked': torch.stack([item['rz_masked'] for item in batch]),
    }
