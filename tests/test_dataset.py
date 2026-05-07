"""
Tests for TokenizedSpectrumDataset
==================================
"""

import torch
import pytest
from pathlib import Path

from src.datasets.tokenized_dataset import TokenizedSpectrumDataset, collate_tokenized_batch
from src.tokenizers.spectrum import SpectrumTokenizer
from src.tokenizers.redshift import RedshiftTokenizer
from src.utils.data import DESISpectrumDataset


class TestTokenizedDataset:
    """Test tokenized dataset."""
    
    @pytest.fixture
    def spectra(self):
        """Load a few real spectra."""
        files = sorted(Path('data/desi_raw').glob('coadd-*.fits'))
        if not files:
            pytest.skip("No data files found")
        
        all_spectra = []
        for f in files[:1]:  # Just first file
            rr = f.parent / f.name.replace('coadd-', 'redrock-')
            ds = DESISpectrumDataset(
                coadd_path=f,
                redrock_path=rr,
                require_good_zwarn=False,
                require_nonzero_flux=True,
            )
            for i in range(min(5, len(ds))):
                all_spectra.append(ds[i])
        
        return all_spectra
    
    @pytest.fixture
    def tokenizers(self, spectra):
        """Create fitted tokenizers."""
        spectrum_tok = SpectrumTokenizer()
        
        all_z = [s['z'] for s in spectra]
        redshift_tok = RedshiftTokenizer(n_levels=256)
        redshift_tok.fit(all_z)
        
        return spectrum_tok, redshift_tok
    
    def test_dataset_length(self, spectra, tokenizers):
        """Dataset length should match number of spectra."""
        spectrum_tok, redshift_tok = tokenizers
        dataset = TokenizedSpectrumDataset(spectra, spectrum_tok, redshift_tok, approach='a')
        assert len(dataset) == len(spectra)
    
    def test_getitem_shapes(self, spectra, tokenizers):
        """getitem should return correct shapes."""
        spectrum_tok, redshift_tok = tokenizers
        dataset = TokenizedSpectrumDataset(spectra, spectrum_tok, redshift_tok, approach='a')
        
        item = dataset[0]
        
        assert 'encoder_input' in item
        assert 'decoder_input' in item
        assert 'target' in item
        assert 'redshift' in item
        assert 'denorm' in item
        
        assert item['encoder_input'].dim() == 1
        assert item['decoder_input'].dim() == 1
        assert item['target'].dim() == 1
        assert item['redshift'].dim() == 0
    
    def test_approach_a_has_redshift_in_encoder(self, spectra, tokenizers):
        """Approach A encoder should contain redshift token."""
        spectrum_tok, redshift_tok = tokenizers
        dataset = TokenizedSpectrumDataset(spectra, spectrum_tok, redshift_tok, approach='a')
        
        item = dataset[0]
        # Redshift token should be in encoder (position 1 after SOS)
        from src.models.transformer import encode_redshift_token
        redshift_idx = redshift_tok.encode(torch.tensor(spectra[0]['z']))
        redshift_token = encode_redshift_token(redshift_idx).item()
        
        assert redshift_token in item['encoder_input']
    
    def test_approach_b_no_redshift_in_encoder(self, spectra, tokenizers):
        """Approach B encoder should NOT contain redshift token."""
        spectrum_tok, redshift_tok = tokenizers
        dataset = TokenizedSpectrumDataset(spectra, spectrum_tok, redshift_tok, approach='b')
        
        item = dataset[0]
        from src.models.transformer import encode_redshift_token
        redshift_idx = redshift_tok.encode(torch.tensor(spectra[0]['z']))
        redshift_token = encode_redshift_token(redshift_idx).item()
        
        assert redshift_token not in item['encoder_input']
    
    def test_decoder_input_starts_with_sos(self, spectra, tokenizers):
        """Decoder input should always start with SOS."""
        spectrum_tok, redshift_tok = tokenizers
        
        for approach in ['a', 'b']:
            dataset = TokenizedSpectrumDataset(spectra, spectrum_tok, redshift_tok, approach=approach)
            item = dataset[0]
            assert item['decoder_input'][0].item() == 0  # SOS_TOKEN
    
    def test_target_length_matches_decoder(self, spectra, tokenizers):
        """Target should be same length as decoder input."""
        spectrum_tok, redshift_tok = tokenizers
        dataset = TokenizedSpectrumDataset(spectra, spectrum_tok, redshift_tok, approach='a')
        
        item = dataset[0]
        assert len(item['target']) == len(item['decoder_input'])
    
    def test_collate_batch(self, spectra, tokenizers):
        """Collate should batch and pad correctly."""
        spectrum_tok, redshift_tok = tokenizers
        dataset = TokenizedSpectrumDataset(spectra, spectrum_tok, redshift_tok, approach='a')
        
        # Get first 3 items
        items = [dataset[i] for i in range(min(3, len(dataset)))]
        batch = collate_tokenized_batch(items)
        
        assert batch['encoder_input'].dim() == 2  # (B, L)
        assert batch['decoder_input'].dim() == 2
        assert batch['target'].dim() == 2
        assert batch['encoder_mask'].dim() == 2
        assert batch['decoder_mask'].dim() == 2
        
        # Check padding
        assert batch['encoder_mask'].dtype == torch.bool
        assert batch['decoder_mask'].dtype == torch.bool
    
    def test_collate_padding_correct(self, spectra, tokenizers):
        """Padding mask should correctly identify padded positions."""
        spectrum_tok, redshift_tok = tokenizers
        dataset = TokenizedSpectrumDataset(spectra, spectrum_tok, redshift_tok, approach='a')
        
        # Get 2 items with likely different lengths
        items = [dataset[0], dataset[1]]
        batch = collate_tokenized_batch(items)
        
        # For each sequence in batch, check that padded positions are masked
        from src.models.transformer import PAD_TOKEN
        for i in range(len(items)):
            seq = batch['encoder_input'][i]
            mask = batch['encoder_mask'][i]
            
            # Padded positions should be PAD_TOKEN and masked
            padded_positions = (seq == PAD_TOKEN)
            assert torch.equal(padded_positions, mask), \
                f"Padding mask mismatch at batch index {i}"
    
    def test_target_has_ignore_index(self, spectra, tokenizers):
        """Target should use -100 for positions that don't need prediction."""
        spectrum_tok, redshift_tok = tokenizers
        dataset = TokenizedSpectrumDataset(spectra, spectrum_tok, redshift_tok, approach='a')
        
        item = dataset[0]
        target = item['target']
        
        # At least some positions should be valid tokens (not all -100)
        assert (target != -100).any(), "Target should have some valid tokens"
    
    def test_approaches_produce_different_encoders(self, spectra, tokenizers):
        """Approach A and B should produce different encoder inputs."""
        spectrum_tok, redshift_tok = tokenizers
        
        dataset_a = TokenizedSpectrumDataset(spectra, spectrum_tok, redshift_tok, approach='a')
        dataset_b = TokenizedSpectrumDataset(spectra, spectrum_tok, redshift_tok, approach='b')
        
        enc_a = dataset_a[0]['encoder_input']
        enc_b = dataset_b[0]['encoder_input']
        
        # Should be different lengths or different content
        assert not torch.equal(enc_a, enc_b), \
            "Approach A and B should produce different encoder inputs"
        
        # Approach A should be longer (has redshift token)
        assert len(enc_a) == len(enc_b) + 1, \
            "Approach A encoder should be 1 token longer (redshift)"
