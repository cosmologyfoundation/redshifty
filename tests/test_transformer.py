"""
Tests for SpectrumTransformer (Encoder-Decoder)
================================================
"""

import torch
import pytest
from src.models.transformer import (
    SpectrumTransformer,
    RMSNorm,
    RotaryEmbedding,
    MultiHeadAttention,
    EncoderBlock,
    DecoderBlock,
    SwiGLU,
    build_approach_a_sequences,
    build_approach_b_sequences,
    build_encoder_input,
    build_decoder_inputs,
    encode_spectrum_token,
    decode_spectrum_token,
    encode_redshift_token,
    decode_redshift_token,
    is_spectrum_token,
    is_redshift_token,
    SOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    REDMASK_TOKEN,
    TOTAL_VOCAB_SIZE,
    SPECTRUM_TOKEN_OFFSET,
    REDSHIFT_TOKEN_OFFSET,
)


class TestRMSNorm:
    def test_output_shape(self):
        x = torch.randn(2, 10, 64)
        norm = RMSNorm(64)
        out = norm(x)
        assert out.shape == x.shape
    
    def test_rms_is_one(self):
        x = torch.randn(1, 5, 32)
        norm = RMSNorm(32)
        out = norm(x)
        rms = torch.sqrt(torch.mean(out ** 2, dim=-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)


class TestRotaryEmbedding:
    def test_shape(self):
        rope = RotaryEmbedding(64, max_seq_len=128)
        cos, sin = rope(32, torch.device('cpu'))
        assert cos.shape == (32, 64)
        assert sin.shape == (32, 64)
    
    def test_rotation_invariance(self):
        rope = RotaryEmbedding(64)
        cos, sin = rope(16, torch.device('cpu'))
        x = torch.randn(1, 1, 16, 64)
        x_rot1 = rope.apply_rotary_pos_emb(x, x.clone(), cos, sin)[0]
        assert not torch.allclose(x, x_rot1)


class TestMultiHeadAttention:
    def test_self_attention_shape(self):
        attn = MultiHeadAttention(128, 4, causal=False, use_rope=True)
        x = torch.randn(2, 16, 128)
        cos = torch.randn(16, 32)
        sin = torch.randn(16, 32)
        out = attn(x, cos=cos, sin=sin)
        assert out.shape == x.shape
    
    def test_cross_attention_shape(self):
        attn = MultiHeadAttention(128, 4, causal=False, use_rope=False)
        x = torch.randn(2, 10, 128)  # queries
        context = torch.randn(2, 20, 128)  # keys/values
        out = attn(x, context=context)
        assert out.shape == x.shape
    
    def test_causal_mask(self):
        attn = MultiHeadAttention(64, 4, causal=True, use_rope=True)
        x = torch.zeros(1, 5, 64)
        x[0, 2, :] = 10.0
        cos = torch.randn(5, 16)
        sin = torch.randn(5, 16)
        out = attn(x, cos=cos, sin=sin)
        assert not torch.allclose(out[0, 0], out[0, 2])


class TestSwiGLU:
    def test_shape(self):
        mlp = SwiGLU(128, hidden_dim=256)
        x = torch.randn(2, 10, 128)
        out = mlp(x)
        assert out.shape == x.shape


class TestEncoderBlock:
    def test_shape(self):
        block = EncoderBlock(128, 4)
        x = torch.randn(2, 16, 128)
        cos = torch.randn(16, 32)
        sin = torch.randn(16, 32)
        out = block(x, cos, sin)
        assert out.shape == x.shape


class TestDecoderBlock:
    def test_shape(self):
        block = DecoderBlock(128, 4)
        x = torch.randn(2, 10, 128)
        enc = torch.randn(2, 20, 128)
        cos = torch.randn(10, 32)
        sin = torch.randn(10, 32)
        out = block(x, enc, cos, sin)
        assert out.shape == x.shape


class TestSpectrumTransformer:
    def test_forward_shape(self):
        model = SpectrumTransformer(
            vocab_size=TOTAL_VOCAB_SIZE,
            d_model=256,
            n_encoder_layers=2,
            n_decoder_layers=2,
            n_heads=4,
            max_seq_len=128,
        )
        enc_ids = torch.randint(0, TOTAL_VOCAB_SIZE, (2, 15))
        dec_ids = torch.randint(0, TOTAL_VOCAB_SIZE, (2, 12))
        logits, loss = model(enc_ids, dec_ids)
        assert logits.shape == (2, 12, TOTAL_VOCAB_SIZE)
        assert loss is None
    
    def test_forward_with_targets(self):
        model = SpectrumTransformer(
            vocab_size=TOTAL_VOCAB_SIZE,
            d_model=256,
            n_encoder_layers=2,
            n_decoder_layers=2,
            n_heads=4,
            max_seq_len=128,
        )
        enc_ids = torch.randint(0, TOTAL_VOCAB_SIZE, (2, 15))
        dec_ids = torch.randint(0, TOTAL_VOCAB_SIZE, (2, 12))
        targets = torch.randint(0, TOTAL_VOCAB_SIZE, (2, 12))
        logits, loss = model(enc_ids, dec_ids, targets=targets)
        assert logits.shape == (2, 12, TOTAL_VOCAB_SIZE)
        assert loss is not None
        assert loss.dim() == 0
    
    def test_forward_with_ignore_index(self):
        model = SpectrumTransformer(
            vocab_size=TOTAL_VOCAB_SIZE,
            d_model=256,
            n_encoder_layers=2,
            n_decoder_layers=2,
            n_heads=4,
            max_seq_len=128,
        )
        enc_ids = torch.randint(0, TOTAL_VOCAB_SIZE, (2, 15))
        dec_ids = torch.randint(0, TOTAL_VOCAB_SIZE, (2, 12))
        targets = torch.randint(0, TOTAL_VOCAB_SIZE, (2, 12))
        targets[0, 0] = -100
        logits, loss = model(enc_ids, dec_ids, targets=targets)
        assert loss is not None
    
    def test_generate(self):
        model = SpectrumTransformer(
            vocab_size=TOTAL_VOCAB_SIZE,
            d_model=256,
            n_encoder_layers=2,
            n_decoder_layers=2,
            n_heads=4,
            max_seq_len=128,
        )
        enc_ids = torch.tensor([[SOS_TOKEN, REDMASK_TOKEN, 10, 20, 30, EOS_TOKEN]])
        output = model.generate(enc_ids, max_new_tokens=5)
        assert output.shape[1] > 1
    
    def test_sequence_length_exceeds_max(self):
        model = SpectrumTransformer(
            vocab_size=TOTAL_VOCAB_SIZE,
            d_model=256,
            n_encoder_layers=2,
            n_decoder_layers=2,
            n_heads=4,
            max_seq_len=10,
        )
        enc_ids = torch.randint(0, TOTAL_VOCAB_SIZE, (1, 20))
        dec_ids = torch.randint(0, TOTAL_VOCAB_SIZE, (1, 5))
        with pytest.raises(AssertionError):
            model(enc_ids, dec_ids)


class TestTokenEncoding:
    def test_spectrum_encode_decode(self):
        lfq_idx = torch.tensor([0, 512, 1023])
        token_ids = encode_spectrum_token(lfq_idx)
        decoded = decode_spectrum_token(token_ids)
        assert torch.equal(lfq_idx, decoded)
        assert (token_ids >= SPECTRUM_TOKEN_OFFSET).all()
        assert (token_ids < REDSHIFT_TOKEN_OFFSET).all()
    
    def test_redshift_encode_decode(self):
        fsq_idx = torch.tensor([0, 128, 255])
        token_ids = encode_redshift_token(fsq_idx)
        decoded = decode_redshift_token(token_ids)
        assert torch.equal(fsq_idx, decoded)
        assert (token_ids >= REDSHIFT_TOKEN_OFFSET).all()
        assert (token_ids < TOTAL_VOCAB_SIZE).all()
    
    def test_is_spectrum_token(self):
        tokens = torch.tensor([0, 8, 500, 1031, 1032, 1287])
        mask = is_spectrum_token(tokens)
        expected = torch.tensor([False, True, True, True, False, False])
        assert torch.equal(mask, expected)
    
    def test_is_redshift_token(self):
        tokens = torch.tensor([0, 8, 1031, 1032, 1100, 1287])
        mask = is_redshift_token(tokens)
        expected = torch.tensor([False, False, False, True, True, True])
        assert torch.equal(mask, expected)


class TestSequenceBuilding:
    def test_build_encoder_with_redshift(self):
        redshift = torch.tensor(1100)
        spectrum = torch.tensor([10, 20, 30])
        enc = build_encoder_input(redshift, spectrum, include_redshift=True)
        assert enc[0] == SOS_TOKEN
        assert enc[1] == 1100
        assert torch.equal(enc[2:5], spectrum)
        assert enc[-1] == EOS_TOKEN
    
    def test_build_encoder_without_redshift(self):
        redshift = torch.tensor(1100)
        spectrum = torch.tensor([10, 20, 30])
        enc = build_encoder_input(redshift, spectrum, include_redshift=False)
        assert enc[0] == SOS_TOKEN
        assert enc[1] == 10  # First spectrum token
        assert enc[-1] == EOS_TOKEN
    
    def test_build_decoder_inputs(self):
        redshift = torch.tensor(1100)
        spectrum = torch.tensor([10, 20, 30])
        dec_in, target = build_decoder_inputs(redshift, spectrum)
        
        # Decoder input: [SOS, redshift, s1, s2, s3]
        assert dec_in[0] == SOS_TOKEN
        assert dec_in[1] == 1100
        assert torch.equal(dec_in[2:], spectrum)
        
        # Target: [redshift, s1, s2, s3, EOS]
        assert target[0] == 1100
        assert torch.equal(target[1:4], spectrum)
        assert target[-1] == EOS_TOKEN
    
    def test_approach_a_sequences(self):
        redshift = torch.tensor(1100)
        spectrum = torch.tensor([10, 20, 30])
        enc, dec_in, target = build_approach_a_sequences(redshift, spectrum)
        
        # Encoder has redshift
        assert enc[1] == 1100
        # Decoder input starts with SOS + redshift
        assert dec_in[0] == SOS_TOKEN
        assert dec_in[1] == 1100
        # Target starts with redshift
        assert target[0] == 1100
    
    def test_approach_b_sequences(self):
        redshift = torch.tensor(1100)
        spectrum = torch.tensor([10, 20, 30])
        enc, dec_in, target = build_approach_b_sequences(redshift, spectrum)
        
        # Encoder does NOT have redshift
        assert enc[1] == 10  # First spectrum token
        assert 1100 not in enc
        
        # Decoder still gets redshift (teacher forcing)
        assert dec_in[1] == 1100
        assert target[0] == 1100
    
    def test_end_to_end_approach_a(self):
        """Test model forward with Approach A sequences."""
        model = SpectrumTransformer(
            vocab_size=TOTAL_VOCAB_SIZE,
            d_model=256,
            n_encoder_layers=2,
            n_decoder_layers=2,
            n_heads=4,
            max_seq_len=128,
        )
        
        redshift = encode_redshift_token(torch.tensor(128))
        spectrum = encode_spectrum_token(torch.tensor([100, 200, 300]))
        
        enc, dec_in, target = build_approach_a_sequences(redshift, spectrum)
        
        logits, loss = model(
            enc.unsqueeze(0),
            dec_in.unsqueeze(0),
            targets=target.unsqueeze(0),
        )
        assert logits.shape == (1, len(dec_in), TOTAL_VOCAB_SIZE)
        assert loss is not None
        assert loss.item() > 0
    
    def test_end_to_end_approach_b(self):
        """Test model forward with Approach B sequences."""
        model = SpectrumTransformer(
            vocab_size=TOTAL_VOCAB_SIZE,
            d_model=256,
            n_encoder_layers=2,
            n_decoder_layers=2,
            n_heads=4,
            max_seq_len=128,
        )
        
        redshift = encode_redshift_token(torch.tensor(128))
        spectrum = encode_spectrum_token(torch.tensor([100, 200, 300]))
        
        enc, dec_in, target = build_approach_b_sequences(redshift, spectrum)
        
        # Verify encoder has no redshift
        assert redshift not in enc
        
        logits, loss = model(
            enc.unsqueeze(0),
            dec_in.unsqueeze(0),
            targets=target.unsqueeze(0),
        )
        assert logits.shape == (1, len(dec_in), TOTAL_VOCAB_SIZE)
        assert loss is not None
        assert loss.item() > 0
    
    def test_model_with_built_sequence(self):
        """Full test: build sequences and run through model."""
        model = SpectrumTransformer(
            vocab_size=TOTAL_VOCAB_SIZE,
            d_model=256,
            n_encoder_layers=2,
            n_decoder_layers=2,
            n_heads=4,
            max_seq_len=128,
        )
        
        redshift = encode_redshift_token(torch.tensor(128))
        spectrum = encode_spectrum_token(torch.tensor([100, 200, 300, 400, 500]))
        
        # Approach A
        enc_a, dec_a, target_a = build_approach_a_sequences(redshift, spectrum)
        logits_a, loss_a = model(enc_a.unsqueeze(0), dec_a.unsqueeze(0), targets=target_a.unsqueeze(0))
        assert logits_a.shape == (1, len(dec_a), TOTAL_VOCAB_SIZE)
        
        # Approach B
        enc_b, dec_b, target_b = build_approach_b_sequences(redshift, spectrum)
        logits_b, loss_b = model(enc_b.unsqueeze(0), dec_b.unsqueeze(0), targets=target_b.unsqueeze(0))
        assert logits_b.shape == (1, len(dec_b), TOTAL_VOCAB_SIZE)
        
        # Both should have loss
        assert loss_a is not None
        assert loss_b is not None
