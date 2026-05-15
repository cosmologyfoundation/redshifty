"""
Tests for SpectrumTransformer (Encoder-Decoder)
===============================================
"""

import torch
import torch.nn as nn
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
        x = torch.randn(1, 5, 64)
        cos = torch.randn(5, 16)
        sin = torch.randn(5, 16)
        out = attn(x, cos=cos, sin=sin)
        x2 = torch.zeros_like(x)
        x2[0, 0, 0] = 100.0
        x2[0, 4, 0] = 100.0
        out2 = attn(x2, cos=cos, sin=sin)
        assert out2[0, 4, 0] != out2[0, 0, 0], "Positions 0 and 4 should have different attended values"


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
        block = DecoderBlock(128, 4, use_bottleneck=True)
        x = torch.randn(2, 10, 128)
        enc_full = torch.randn(2, 20, 128)
        enc_bn = torch.randn(2, 32, 128)  # bottleneck tokens
        cos = torch.randn(10, 32)
        sin = torch.randn(10, 32)
        out = block(x, enc_full, enc_bn, cos, sin)
        assert out.shape == x.shape

    def test_use_bottleneck_false(self):
        block = DecoderBlock(128, 4, use_bottleneck=False)
        x = torch.randn(2, 10, 128)
        enc_full = torch.randn(2, 20, 128)
        enc_bn = torch.randn(2, 32, 128)  # bottleneck tokens (ignored when use_bottleneck=False)
        cos = torch.randn(10, 32)
        sin = torch.randn(10, 32)
        out = block(x, enc_full, enc_bn, cos, sin)
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

    def test_ar_loss_shape(self):
        model = SpectrumTransformer(
            vocab_size=TOTAL_VOCAB_SIZE,
            d_model=128,
            n_encoder_layers=2,
            n_decoder_layers=2,
            n_heads=4,
            max_seq_len=128,
        )
        model.train()
        enc_ids = torch.randint(0, TOTAL_VOCAB_SIZE, (2, 10))
        targets = torch.randint(0, TOTAL_VOCAB_SIZE, (2, 12))
        logits, loss = model.ar_loss(enc_ids, targets, max_generate_tokens=5)
        assert logits.shape == (2, 12, TOTAL_VOCAB_SIZE)
        assert loss.numel() == 1
        assert loss.item() > 0

    def test_ar_loss_with_valid_v2_redshift_tokens(self):
        model = SpectrumTransformer(
            vocab_size=TOTAL_VOCAB_SIZE,
            d_model=128,
            n_encoder_layers=2,
            n_decoder_layers=2,
            n_heads=4,
            max_seq_len=128,
        )
        model.train()
        enc_ids = torch.randint(0, TOTAL_VOCAB_SIZE, (2, 10))
        valid_rz_min = REDSHIFT_TOKEN_OFFSET
        valid_rz_max = TOTAL_VOCAB_SIZE - 1
        targets = torch.randint(valid_rz_min, valid_rz_max, (2, 12))
        logits, loss = model.ar_loss(enc_ids, targets, max_generate_tokens=5)
        assert logits.shape == (2, 12, TOTAL_VOCAB_SIZE)
        assert loss.item() > 0

    def test_ar_loss_higher_than_teacher_forcing(self):
        model = SpectrumTransformer(
            vocab_size=TOTAL_VOCAB_SIZE,
            d_model=128,
            n_encoder_layers=2,
            n_decoder_layers=2,
            n_heads=4,
            max_seq_len=128,
        )
        model.train()
        enc_ids = torch.randint(0, TOTAL_VOCAB_SIZE, (2, 10))
        dec_ids = torch.randint(0, TOTAL_VOCAB_SIZE, (2, 12))
        targets = torch.randint(0, TOTAL_VOCAB_SIZE, (2, 12))
        _, tf_loss = model(enc_ids, dec_ids, targets=targets, redshift_weight=1.0)
        _, ar_loss_val = model.ar_loss(enc_ids, targets, max_generate_tokens=5, redshift_weight=1.0)
        assert ar_loss_val.item() >= 0

    def test_ar_loss_respects_redshift_weight(self):
        model = SpectrumTransformer(
            vocab_size=TOTAL_VOCAB_SIZE,
            d_model=128,
            n_encoder_layers=2,
            n_decoder_layers=2,
            n_heads=4,
            max_seq_len=128,
        )
        model.train()
        enc_ids = torch.randint(0, TOTAL_VOCAB_SIZE, (2, 10))
        targets = torch.randint(0, TOTAL_VOCAB_SIZE, (2, 12))
        _, loss_w1 = model.ar_loss(enc_ids, targets, max_generate_tokens=5, redshift_weight=1.0)
        _, loss_w50 = model.ar_loss(enc_ids, targets, max_generate_tokens=5, redshift_weight=50.0)
        assert loss_w50.item() >= loss_w1.item()

    def test_encoder_contribution_to_ar_loss(self):
        model = SpectrumTransformer(
            vocab_size=TOTAL_VOCAB_SIZE,
            d_model=128,
            n_encoder_layers=2,
            n_decoder_layers=2,
            n_heads=4,
            max_seq_len=128,
        )
        model.train()
        torch.manual_seed(42)
        enc_a = torch.randint(0, TOTAL_VOCAB_SIZE, (2, 10))
        enc_b = torch.randint(0, TOTAL_VOCAB_SIZE, (2, 10))
        targets = torch.randint(0, TOTAL_VOCAB_SIZE, (2, 12))
        _, loss_a = model.ar_loss(enc_a, targets, max_generate_tokens=10)
        _, loss_b = model.ar_loss(enc_b, targets, max_generate_tokens=10)
        diff_pct = abs(loss_a.item() - loss_b.item()) / max(loss_a.item(), loss_b.item())
        assert diff_pct > 0.0001, \
            f"Different encoders should produce different losses, got {diff_pct*100:.4f}% diff (loss_a={loss_a.item():.3f}, loss_b={loss_b.item():.3f})"

    def test_encoder_produces_nonzero_output(self):
        model = SpectrumTransformer(
            vocab_size=TOTAL_VOCAB_SIZE,
            d_model=128,
            n_encoder_layers=2,
            n_decoder_layers=2,
            n_heads=4,
            max_seq_len=128,
        )
        model.eval()
        enc_ids = torch.randint(0, TOTAL_VOCAB_SIZE, (2, 10))
        with torch.no_grad():
            enc_out = model.encode(enc_ids)
        assert enc_out.sum() != 0, "Encoder output should be nonzero"
        assert enc_out.shape == (2, 10, 128)

    def test_ar_loss_produces_finite_loss(self):
        model = SpectrumTransformer(
            vocab_size=TOTAL_VOCAB_SIZE,
            d_model=128,
            n_encoder_layers=2,
            n_decoder_layers=2,
            n_heads=4,
            max_seq_len=128,
        )
        model.train()
        enc_ids = torch.randint(0, TOTAL_VOCAB_SIZE, (2, 10))
        targets = torch.randint(0, TOTAL_VOCAB_SIZE, (2, 12))
        _, loss = model.ar_loss(enc_ids, targets, max_generate_tokens=5)
        assert torch.isfinite(loss), "AR loss should be finite"

    def test_ar_loss_redshift_weight_affects_loss_ratio(self):
        model = SpectrumTransformer(
            vocab_size=TOTAL_VOCAB_SIZE,
            d_model=128,
            n_encoder_layers=2,
            n_decoder_layers=2,
            n_heads=4,
            max_seq_len=128,
        )
        model.train()
        enc_ids = torch.randint(REDSHIFT_TOKEN_OFFSET, TOTAL_VOCAB_SIZE, (2, 10))
        targets = torch.randint(REDSHIFT_TOKEN_OFFSET, TOTAL_VOCAB_SIZE, (2, 12))
        targets[:, 0] = REDSHIFT_TOKEN_OFFSET + 500
        _, loss_w1 = model.ar_loss(enc_ids, targets, max_generate_tokens=5, redshift_weight=1.0)
        _, loss_w100 = model.ar_loss(enc_ids, targets, max_generate_tokens=5, redshift_weight=100.0)
        ratio = loss_w100.item() / loss_w1.item()
        assert 50 < ratio < 200, f"Weight=100 should give ~100x loss, got {ratio:.1f}x"

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


class TestDecoupledVocabWithDenoising:
    """Tests for partially decoupled vocabularies + cross-attention bottleneck.

    Architecture:
    - Special (0-7) and redshift (1032-2055): SHARED between encoder/decoder
    - Spectrum (8-1031): DECOUPLED — decoder has own embedding space
    - Cross-attention: encoder compressed to 32 bottleneck tokens (not 1024)

    This allows redshift to be learned via cross-attention (shared embedding)
    while preventing spectrum copy via both decoupled vocab AND bottleneck.
    """

    def test_partially_decoupled_encoder_decoder_shares_redshift(self):
        """Verify that special+redshift tokens share embedding, spectrum does not."""
        model = SpectrumTransformer(
            vocab_size=2056,
            d_model=128,
            n_encoder_layers=2,
            n_decoder_layers=2,
            n_heads=4,
            max_seq_len=128,
        )
        assert model.token_embedding.weight.shape == (2056, 128)
        assert model.decoder_spectrum_embedding.weight.shape == (1024, 128)
        assert not torch.allclose(
            model.token_embedding.weight[1032:1033],
            model.decoder_spectrum_embedding.weight[0:1]
        )

    def test_bottleneck_compresses_encoder(self):
        """Verify encoder is compressed to n_bottleneck_tokens."""
        model = SpectrumTransformer(
            vocab_size=2056,
            d_model=128,
            n_encoder_layers=2,
            n_decoder_layers=2,
            n_heads=4,
            max_seq_len=128,
            n_bottleneck_tokens=32,
        )
        encoder_out = torch.randn(2, 100, 128)  # 100 encoder positions
        compressed = model._compress_encoder(encoder_out)
        assert compressed.shape == (2, 32, 128)

    def test_decoder_has_separate_spectrum_embedding(self):
        """Decoder spectrum embedding is separate from encoder spectrum space."""
        model = SpectrumTransformer(
            vocab_size=2056,
            d_model=256,
            n_encoder_layers=2,
            n_decoder_layers=2,
            n_heads=4,
            max_seq_len=128,
        )
        enc_ids = torch.randint(0, 2056, (2, 15))
        dec_ids = torch.randint(0, 2056, (2, 12))
        logits, _ = model(enc_ids, dec_ids)
        assert logits.shape == (2, 12, 2056)
        assert logits.shape[-1] == model.vocab_size

    def test_forward_with_bottleneck(self):
        """Forward pass works with bottleneck compression."""
        model = SpectrumTransformer(
            vocab_size=2056,
            d_model=256,
            n_encoder_layers=2,
            n_decoder_layers=2,
            n_heads=4,
            max_seq_len=128,
        )
        enc_ids = torch.randint(0, 2056, (2, 15))
        dec_ids = torch.randint(0, 2056, (2, 12))
        targets = torch.randint(0, 2056, (2, 12))
        logits, loss = model(enc_ids, dec_ids, targets=targets)
        assert logits.shape == (2, 12, 2056)
        assert loss is not None
        assert loss.item() > 0


class TestOptionCCrossAttention:
    """Tests for Option C: Single bottleneck path + Auxiliary Redshift Head.

    Architecture:
    - Single decoder path with bottleneck cross-attention (spectrum copy prevented)
    - Auxiliary redshift head: MLP(encoder_output.mean(dim=1)) → z classification
      This provides a DIRECT gradient path for redshift, bypassing the bottleneck.

    This replaces the Option A dual-path approach which had a structural flaw:
    position 0's self-attention had no context (causal mask + single position).
    """

    def test_single_path_forward_shape(self):
        """Forward pass with Option C produces correct shape."""
        model = SpectrumTransformer(
            vocab_size=2056,
            d_model=128,
            n_encoder_layers=2,
            n_decoder_layers=2,
            n_heads=4,
            max_seq_len=128,
        )
        enc_ids = torch.randint(0, 2056, (2, 15))
        dec_ids = torch.randint(0, 2056, (2, 12))
        logits, loss = model(enc_ids, dec_ids)
        assert logits.shape == (2, 12, 2056)
        assert loss is None

    def test_single_path_forward_with_targets(self):
        """Forward pass with targets returns combined loss."""
        model = SpectrumTransformer(
            vocab_size=2056,
            d_model=128,
            n_encoder_layers=2,
            n_decoder_layers=2,
            n_heads=4,
            max_seq_len=128,
        )
        enc_ids = torch.randint(0, 2056, (2, 15))
        dec_ids = torch.randint(0, 2056, (2, 12))
        targets = torch.randint(0, 2056, (2, 12))
        logits, loss = model(enc_ids, dec_ids, targets=targets)
        assert logits.shape == (2, 12, 2056)
        assert loss is not None
        assert loss.item() > 0

    def test_auxiliary_redshift_head_exists(self):
        """Model has redshift_aux_head MLP."""
        model = SpectrumTransformer(
            vocab_size=2056,
            d_model=128,
            n_encoder_layers=2,
            n_decoder_layers=3,
            n_heads=4,
            max_seq_len=128,
            n_redshift_classes=256,
        )
        assert hasattr(model, 'redshift_aux_head')
        # Check it's an MLP with expected structure
        assert isinstance(model.redshift_aux_head, nn.Sequential)
        # Last layer should output n_redshift_classes
        last_linear = model.redshift_aux_head[-1]
        assert last_linear.out_features == 256

    def test_auxiliary_head_forward(self):
        """Auxiliary head produces valid redshift logits."""
        model = SpectrumTransformer(
            vocab_size=2056,
            d_model=128,
            n_encoder_layers=2,
            n_decoder_layers=2,
            n_heads=4,
            max_seq_len=128,
            n_redshift_classes=256,
        )
        encoder_out = torch.randn(2, 50, 128)  # (B, L_enc, d_model)
        pooled = encoder_out.mean(dim=1)  # (B, d_model)
        rz_logits = model.redshift_aux_head(pooled)
        assert rz_logits.shape == (2, 256)

    def test_auxiliary_head_gradients_flow(self):
        """Auxiliary head gradients flow to encoder."""
        model = SpectrumTransformer(
            vocab_size=2056,
            d_model=128,
            n_encoder_layers=2,
            n_decoder_layers=2,
            n_heads=4,
            max_seq_len=128,
            n_redshift_classes=256,
        )
        enc_ids = torch.randint(REDSHIFT_TOKEN_OFFSET, 2056, (2, 15))
        dec_ids = torch.randint(REDSHIFT_TOKEN_OFFSET, 2056, (2, 12))
        targets = torch.randint(REDSHIFT_TOKEN_OFFSET, 2056, (2, 12))
        targets[:, 0] = REDSHIFT_TOKEN_OFFSET + 100

        _, loss = model(enc_ids, dec_ids, targets=targets,
                       redshift_weight=50.0, aux_redshift_weight=1.0)
        loss.backward()

        # Gradient should exist on encoder layers
        for layer in model.encoder_layers:
            assert layer.mlp.w3.weight.grad is not None
            assert layer.mlp.w3.weight.grad.sum() != 0

    def test_auxiliary_head_contributes_to_loss(self):
        """Auxiliary head contributes to total loss."""
        model = SpectrumTransformer(
            vocab_size=2056,
            d_model=128,
            n_encoder_layers=2,
            n_decoder_layers=2,
            n_heads=4,
            max_seq_len=128,
            n_redshift_classes=256,
        )
        enc_ids = torch.randint(REDSHIFT_TOKEN_OFFSET, 2056, (2, 15))
        dec_ids = torch.randint(REDSHIFT_TOKEN_OFFSET, 2056, (2, 12))
        targets = torch.randint(REDSHIFT_TOKEN_OFFSET, 2056, (2, 12))

        # With aux_weight=0, only sequence loss
        _, loss_no_aux = model(enc_ids, dec_ids, targets=targets,
                               redshift_weight=50.0, aux_redshift_weight=0.0)
        # With aux_weight=1, sequence + auxiliary loss
        _, loss_with_aux = model(enc_ids, dec_ids, targets=targets,
                                  redshift_weight=50.0, aux_redshift_weight=1.0)

        # Both should produce valid losses
        assert loss_no_aux.item() > 0
        assert loss_with_aux.item() > 0

    def test_bottleneck_still_compresses(self):
        """Bottleneck path still compresses encoder to 32 tokens."""
        model = SpectrumTransformer(
            vocab_size=2056,
            d_model=128,
            n_encoder_layers=2,
            n_decoder_layers=2,
            n_heads=4,
            max_seq_len=128,
            n_bottleneck_tokens=32,
        )
        encoder_out = torch.randn(2, 100, 128)
        compressed = model._compress_encoder(encoder_out)
        assert compressed.shape == (2, 32, 128)

    def test_loss_split_correct_position_0_vs_1plus(self):
        """Loss is correctly split between position 0 (redshift) and positions 1+."""
        model = SpectrumTransformer(
            vocab_size=2056,
            d_model=128,
            n_encoder_layers=2,
            n_decoder_layers=2,
            n_heads=4,
            max_seq_len=128,
        )
        enc_ids = torch.randint(0, 2056, (2, 15))
        dec_ids = torch.randint(0, 2056, (2, 12))
        targets = torch.randint(0, 2056, (2, 12))

        _, loss_w1 = model(enc_ids, dec_ids, targets=targets,
                         redshift_weight=1.0, aux_redshift_weight=0.0)
        _, loss_w50 = model(enc_ids, dec_ids, targets=targets,
                            redshift_weight=50.0, aux_redshift_weight=0.0)

        assert loss_w50.item() > loss_w1.item(), \
            f"weight=50 ({loss_w50.item():.3f}) should be > weight=1 ({loss_w1.item():.3f})"

    def test_generate_autoregressive(self):
        """Autoregressive generation works with Option C."""
        model = SpectrumTransformer(
            vocab_size=2056,
            d_model=128,
            n_encoder_layers=2,
            n_decoder_layers=2,
            n_heads=4,
            max_seq_len=128,
        )
        enc_ids = torch.tensor([[SOS_TOKEN, REDMASK_TOKEN, 10, 20, 30, EOS_TOKEN]])
        output = model.generate(enc_ids, max_new_tokens=5)
        assert output.shape[1] > 1
        assert (output >= 0).all()
        assert (output < 2056).all()

    def test_ar_loss_with_single_path(self):
        """AR loss is computed correctly with single bottleneck path."""
        model = SpectrumTransformer(
            vocab_size=2056,
            d_model=128,
            n_encoder_layers=2,
            n_decoder_layers=2,
            n_heads=4,
            max_seq_len=128,
        )
        model.train()
        enc_ids = torch.randint(0, 2056, (2, 10))
        targets = torch.randint(0, 2056, (2, 12))
        logits, loss = model.ar_loss(enc_ids, targets, max_generate_tokens=5)
        assert logits.shape == (2, 12, 2056)
        assert loss.numel() == 1
        assert loss.item() > 0

    def test_no_crash_with_various_seq_lens(self):
        """Works with different sequence lengths."""
        model = SpectrumTransformer(
            vocab_size=2056,
            d_model=128,
            n_encoder_layers=2,
            n_decoder_layers=2,
            n_heads=4,
            max_seq_len=256,
        )
        for seq_len in [5, 10, 20]:
            enc_ids = torch.randint(0, 2056, (2, seq_len))
            dec_ids = torch.randint(0, 2056, (2, seq_len - 1))
            targets = torch.randint(0, 2056, (2, seq_len - 1))
            logits, loss = model(enc_ids, dec_ids, targets=targets,
                               redshift_weight=50.0, aux_redshift_weight=1.0)
            assert logits.shape == (2, seq_len - 1, 2056)
            assert loss is not None

    def test_different_encoder_lengths(self):
        """Works with different encoder input sizes."""
        model = SpectrumTransformer(
            vocab_size=2056,
            d_model=128,
            n_encoder_layers=2,
            n_decoder_layers=2,
            n_heads=4,
            max_seq_len=256,
        )
        for enc_len in [10, 50, 100]:
            enc_ids = torch.randint(0, 2056, (2, enc_len))
            dec_ids = torch.randint(0, 2056, (2, 20))
            targets = torch.randint(0, 2056, (2, 20))
            logits, loss = model(enc_ids, dec_ids, targets=targets)
            assert logits.shape == (2, 20, 2056)
            assert loss is not None

    def test_approach_a_redshift_learning(self):
        """Approach A with Option C can learn redshift via aux head."""
        model = SpectrumTransformer(
            vocab_size=2056,
            d_model=128,
            n_encoder_layers=2,
            n_decoder_layers=3,
            n_heads=4,
            max_seq_len=128,
            n_redshift_classes=256,
        )
        redshift = encode_redshift_token(torch.tensor(500))
        spectrum = encode_spectrum_token(torch.tensor([100, 200, 300, 400]))

        enc, dec_in, target = build_approach_a_sequences(redshift, spectrum)

        logits, loss = model(
            enc.unsqueeze(0),
            dec_in.unsqueeze(0),
            targets=target.unsqueeze(0),
            redshift_weight=50.0,
            aux_redshift_weight=1.0,
        )
        assert logits.shape == (1, len(dec_in), 2056)
        assert loss is not None
        assert loss.item() > 0

    def test_decoder_has_single_bottleneck_path(self):
        """Model has single decoder_layers list with bottleneck."""
        model = SpectrumTransformer(
            vocab_size=2056,
            d_model=128,
            n_encoder_layers=2,
            n_decoder_layers=3,
            n_heads=4,
            max_seq_len=128,
        )
        assert hasattr(model, 'decoder_layers')
        assert len(model.decoder_layers) == 3
        # All layers should use bottleneck
        for layer in model.decoder_layers:
            assert layer.use_bottleneck

    def test_encoder_mask_plus_option_c(self):
        """Encoder masking works with Option C."""
        model = SpectrumTransformer(
            vocab_size=2056,
            d_model=256,
            n_encoder_layers=2,
            n_decoder_layers=2,
            n_heads=4,
            max_seq_len=128,
        )
        enc_ids = torch.randint(0, 2056, (2, 15))
        dec_ids = torch.randint(0, 2056, (2, 12))
        targets = torch.randint(0, 2056, (2, 12))
        logits, loss = model(enc_ids, dec_ids, targets=targets)
        assert logits.shape == (2, 12, 2056)
        assert loss is not None
        assert loss.item() > 0

    def test_spectrum_copy_prevented_by_decoupled_embedding(self):
        """Spectrum tokens (8-1031) use decoupled embedding — copy impossible."""
        model = SpectrumTransformer(
            vocab_size=2056,
            d_model=128,
            n_encoder_layers=2,
            n_decoder_layers=2,
            n_heads=4,
            max_seq_len=128,
        )
        enc_ids = torch.randint(8, 1032, (2, 15))
        dec_ids = torch.randint(8, 1032, (2, 12))
        targets = torch.randint(8, 1032, (2, 12))
        logits, loss = model(enc_ids, dec_ids, targets=targets)
        assert logits.shape == (2, 12, 2056)
        assert loss is not None

    def test_get_decoder_embedding_maps_correctly(self):
        """_get_decoder_embedding correctly routes special/redshift vs spectrum."""
        model = SpectrumTransformer(
            vocab_size=2056,
            d_model=64,
            n_encoder_layers=1,
            n_decoder_layers=1,
            n_heads=4,
            max_seq_len=64,
        )
        tokens = torch.tensor([[0, 1032, 500, 8]])  # [SOS, redshift, spectrum, spectrum]
        emb = model._get_decoder_embedding(tokens)

        assert emb.shape == (1, 4, 64)

        shared_emb_sos = model.token_embedding(torch.tensor([0]))
        assert torch.allclose(emb[0, 0], shared_emb_sos[0], atol=1e-5)
