"""
Spectrum Transformer (Encoder-Decoder)
======================================
Encoder-decoder transformer for spectrum + redshift token modeling.

Architecture:
- Encoder: Bidirectional self-attention, processes input sequence
- Decoder: Causal self-attention + cross-attention to encoder, generates output
- Both use RoPE, RMSNorm, SwiGLU

Why encoder-decoder for unimodal spectra?
- Approach A: Encoder sees redshift + spectrum bidirectionally
- Approach B: Encoder sees spectrum ONLY (no redshift). Decoder predicts redshift
  from cross-attention, then generates spectrum conditioned on predicted redshift.
  This avoids teacher-forcing leakage in decoder-only models.

Special Tokens:
  0: [SOS]      - Start of sequence
  1: [EOS]      - End of sequence
  2: [PAD]      - Padding
  3: [MASK]     - Masked spectrum token
  4: [REDMASK]  - Masked redshift token
  5: [SPEC_SEP] - Spectrum separator (reserved)
  6-7: Reserved

Token Offsets:
  Spectrum tokens:  LFQ_index + 8        (range: 8-1031)
  Redshift tokens:  FSQ_index + 1032     (range: 1032-1287)
  Total vocab size: 2056
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# Special token IDs
SOS_TOKEN = 0
EOS_TOKEN = 1
PAD_TOKEN = 2
MASK_TOKEN = 3
REDMASK_TOKEN = 4
SPEC_SEP_TOKEN = 5

# Token offsets
SPECTRUM_TOKEN_OFFSET = 8
REDSHIFT_TOKEN_OFFSET = 1032
TOTAL_VOCAB_SIZE = 2056


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x = x / rms
        return x * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embeddings (RoPE)."""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._precompute(max_seq_len)
    
    def _precompute(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)
    
    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor,
                             cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, L, head_dim)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q_embed = q * cos + self._rotate_half(q) * sin
        k_embed = k * cos + self._rotate_half(k) * sin
        return q_embed, k_embed
    
    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            self._precompute(seq_len)
            self.max_seq_len = seq_len
        return self.cos[:seq_len], self.sin[:seq_len]


class MultiHeadAttention(nn.Module):
    """Multi-head attention supporting self-attention and cross-attention.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout rate
        causal: If True, apply causal mask (for decoder self-attention)
        use_rope: If True, apply RoPE to Q and K
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0,
                 causal: bool = False, use_rope: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.causal = causal
        self.use_rope = use_rope
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
        if use_rope:
            self._rope = RotaryEmbedding(self.head_dim, max_seq_len=2048)
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None,
                cos: Optional[torch.Tensor] = None, sin: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: (B, L, D) query input
            context: (B, L_ctx, D) key/value input (for cross-attention)
            cos, sin: RoPE embeddings (required if use_rope=True)
            mask: (B, L) optional padding mask
            
        Returns:
            out: (B, L, D)
        """
        B, L, D = x.shape
        
        # For cross-attention, K/V come from context
        kv_input = context if context is not None else x
        
        # Project
        q = self.q_proj(x)
        k = self.k_proj(kv_input)
        v = self.v_proj(kv_input)
        
        # Reshape to (B, H, L, head_dim)
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to Q and K (only for self-attention)
        if self.use_rope and cos is not None and sin is not None:
            q, k = self._rope.apply_rotary_pos_emb(q, k, cos, sin)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, L, L_ctx)
        
        # Causal mask
        if self.causal:
            L_ctx = scores.shape[-1]
            causal_mask = torch.triu(torch.ones(L, L_ctx, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Optional padding mask
        if mask is not None:
            if context is not None:
                # Cross-attention: mask applies to context (keys)
                scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            else:
                # Self-attention: mask applies to keys
                scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        # Softmax and apply
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.o_proj(out)
        
        return out


class SwiGLU(nn.Module):
    """SwiGLU MLP activation."""
    
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(8 * dim / 3)
            hidden_dim = ((hidden_dim + 255) // 256) * 256
        
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w1(x))
        up = self.w2(x)
        return self.w3(self.dropout(gate * up))


class EncoderBlock(nn.Module):
    """Encoder block: bidirectional self-attention + MLP."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, causal=False, use_rope=True)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, dropout=dropout)
    
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos=cos, sin=sin, mask=mask)
        x = x + self.mlp(self.norm2(x))
        return x


class DecoderBlock(nn.Module):
    """Decoder block: causal self-attention + cross-attention + MLP.

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout rate
        use_bottleneck: If True, cross-attention uses bottleneck (32 tokens).
            If False, cross-attention uses full encoder output.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0,
                 use_bottleneck: bool = True):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout, causal=True, use_rope=True)
        self.norm2 = RMSNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout, causal=False, use_rope=False)
        self.norm3 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, dropout=dropout)
        self.use_bottleneck = use_bottleneck

    def forward(self, x: torch.Tensor, encoder_full: torch.Tensor,
                encoder_compressed: torch.Tensor,
                self_cos: torch.Tensor, self_sin: torch.Tensor,
                self_mask: Optional[torch.Tensor] = None,
                cross_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Causal self-attention
        x = x + self.self_attn(self.norm1(x), cos=self_cos, sin=self_sin, mask=self_mask)
        # Cross-attention: use bottleneck if use_bottleneck=True, else full encoder
        enc_ctx = encoder_compressed if self.use_bottleneck else encoder_full
        x = x + self.cross_attn(self.norm2(x), context=enc_ctx, mask=cross_mask)
        # MLP
        x = x + self.mlp(self.norm3(x))
        return x


class SpectrumTransformer(nn.Module):
    """Encoder-decoder transformer for spectrum + redshift token modeling.

    Architecture: PARTIALLY decoupled vocabularies + CROSS-ATTENTION BOTTLENECK.

    Key copy-prevention mechanisms:
    1. Partially decoupled vocab: spectrum tokens use separate embedding space
    2. Cross-attention bottleneck: encoder output compressed to 32 tokens via
       learned attention pooling before cross-attention. This removes fine-grained
       positional identity — decoder cannot look up specific encoder positions.
    3. Encoder masking (50%+) further forces the encoder to build robust features

    - Special (0-7) and redshift (1032-2055): SHARED embedding between enc/dec
    - Spectrum (8-1031): DECOUPLED — decoder has own embedding space
    - Cross-attention receives only 32 compressed tokens from encoder, not 1024

    Args:
        vocab_size: Encoder vocabulary size (default 2056)
        d_model: Model dimension (default 768)
        n_encoder_layers: Number of encoder layers (default 6)
        n_decoder_layers: Number of decoder layers (default 6)
        n_heads: Number of attention heads (default 12)
        max_seq_len: Maximum sequence length (default 512)
        dropout: Dropout rate (default 0.1)
        n_bottleneck_tokens: Number of tokens to compress encoder to (default 32)
    """

    SPECTRUM_ENCODER_START = 8
    SPECTRUM_ENCODER_END = 1032
    REDSHIFT_START = 1032
    REDSHIFT_END = 2056

    def __init__(
        self,
        vocab_size: int = TOTAL_VOCAB_SIZE,
        d_model: int = 768,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        n_heads: int = 12,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        n_bottleneck_tokens: int = 32,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.n_bottleneck_tokens = n_bottleneck_tokens

        # Encoder token embeddings (shared with frozen tokenizer)
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Decoder's spectrum embeddings (DECOUPLED from encoder)
        self.decoder_spectrum_embedding = nn.Embedding(1024, d_model)

        # Cross-attention bottleneck: compress encoder to n_bottleneck_tokens
        # 32 tokens forces high-level aggregation — prevents fine-grained position copy
        self.encoder_bottleneck_query = nn.Parameter(torch.randn(1, n_bottleneck_tokens, d_model) * 0.02)
        self.bottleneck_proj = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.LayerNorm(d_model),
        )

        # Shared head for special (0-7) + redshift (1032-2055) tokens
        self.shared_lm_head = nn.Linear(d_model, self.REDSHIFT_END, bias=False)
        self.shared_lm_head.weight = self.token_embedding.weight

        # Separate head for decoder spectrum tokens
        self.spectrum_lm_head = nn.Linear(d_model, 1024, bias=False)

        # RoPE (shared for encoder and decoder self-attention)
        self.rope = RotaryEmbedding(d_model // n_heads, max_seq_len)

        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(d_model, n_heads, dropout)
            for _ in range(n_encoder_layers)
        ])
        self.encoder_norm = RMSNorm(d_model)

        # Decoder: TWO paths for Option A dual cross-attention
        # - decoder_layers_full: for position 0 (redshift) — attends to FULL encoder
        # - decoder_layers_bottleneck: for positions 1+ (spectrum) — attends to 32 tokens
        self.decoder_layers_full = nn.ModuleList([
            DecoderBlock(d_model, n_heads, dropout, use_bottleneck=False)
            for _ in range(n_decoder_layers)
        ])
        self.decoder_layers_bottleneck = nn.ModuleList([
            DecoderBlock(d_model, n_heads, dropout, use_bottleneck=True)
            for _ in range(n_decoder_layers)
        ])
        self.decoder_norm = RMSNorm(d_model)

        self._init_weights()

    def _compress_encoder(self, encoder_out: torch.Tensor) -> torch.Tensor:
        """Compress encoder output to bottleneck tokens via attention pooling.

        Instead of letting the decoder attend to all 1024 encoder positions,
        we compress to n_bottleneck_tokens (default 32). This removes fine-grained
        positional identity — the decoder cannot look up specific encoder positions.
        """
        B, L_enc, D = encoder_out.shape

        # Learnable query tokens for compression
        query = self.encoder_bottleneck_query.expand(B, -1, -1)  # (B, n_bottleneck, D)

        # Simple attention: query attends to encoder to get compressed representation
        # scores: (B, n_bottleneck, L_enc)
        scores = torch.matmul(query, encoder_out.transpose(-2, -1)) / (D ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)

        # Compressed encoder representation: (B, n_bottleneck, D)
        compressed = torch.matmul(attn_weights, encoder_out)
        compressed = self.bottleneck_proj(compressed)

        return compressed

    def _get_decoder_embedding(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map decoder token IDs to embeddings.

        Special tokens (0-7) and redshift tokens (1032-2055) use shared embedding.
        Spectrum tokens (8-1031) use decoder's own spectrum embedding, remapped to
        decoder's local spectrum index (0-1023).
        """
        B, L = token_ids.shape
        emb = torch.zeros(B, L, self.d_model, device=token_ids.device, dtype=self.token_embedding.weight.dtype)

        # Special tokens (0-7) — use shared embedding
        special_mask = token_ids < 8
        emb[special_mask] = self.token_embedding(token_ids[special_mask])

        # Redshift tokens (1032-2055) — use shared embedding
        redshift_mask = (token_ids >= self.REDSHIFT_START) & (token_ids < self.REDSHIFT_END)
        emb[redshift_mask] = self.token_embedding(token_ids[redshift_mask])

        # Spectrum tokens (8-1031) — use decoder's own embedding, remap to [0, 1023]
        spectrum_mask = (token_ids >= self.SPECTRUM_ENCODER_START) & (token_ids < self.SPECTRUM_ENCODER_END)
        if spectrum_mask.any():
            local_idx = token_ids[spectrum_mask] - self.SPECTRUM_ENCODER_START  # maps 8-1031 -> 0-1023
            emb[spectrum_mask] = self.decoder_spectrum_embedding(local_idx)

        return emb

    def _get_decoder_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Compute logits over decoder's vocabulary (2056 tokens).

        Special (0-7) and redshift (1032-2055) logits from shared_lm_head.
        Spectrum (8-1031) logits from spectrum_lm_head, remapped to encoder's spectrum range.
        """
        B, L, D = hidden.shape

        # Logits for special + redshift from shared head
        logits = torch.zeros(B, L, self.vocab_size, device=hidden.device, dtype=hidden.dtype)
        logits[:, :, :self.REDSHIFT_END] = self.shared_lm_head(hidden)

        # Logits for spectrum tokens from decoder's spectrum head, remapped to encoder range
        spectrum_logits = self.spectrum_lm_head(hidden)  # (B, L, 1024)
        logits[:, :, self.SPECTRUM_ENCODER_START:self.SPECTRUM_ENCODER_END] = spectrum_logits

        return logits

    def decode(self, decoder_input_ids: torch.Tensor, encoder_out: torch.Tensor,
               decoder_mask: Optional[torch.Tensor] = None,
               encoder_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode with OPTION A dual cross-attention paths.

        Position 0 (redshift): attends to FULL encoder output via decoder_layers_full.
        Positions 1+ (spectrum): attends to 32-token bottleneck via decoder_layers_bottleneck.

        Args:
            decoder_input_ids: (B, L_dec) decoder input token indices
            encoder_out: (B, L_enc, D) encoder output
            decoder_mask: (B, L_dec) optional padding mask
            encoder_mask: (B, L_enc) optional encoder padding mask

        Returns:
            logits: (B, L_dec, vocab_size=2056)
        """
        x = self._get_decoder_embedding(decoder_input_ids)

        L_dec = decoder_input_ids.shape[1]
        cos, sin = self.rope(L_dec, x.device)

        # Compress encoder to bottleneck for spectrum path
        encoder_compressed = self._compress_encoder(encoder_out)

        # Process through both paths and concatenate
        x_full = x.clone()
        x_bottleneck = x.clone()

        for layer_full, layer_bn in zip(self.decoder_layers_full, self.decoder_layers_bottleneck):
            x_full = layer_full(x_full, encoder_out, encoder_compressed, cos, sin, decoder_mask, None)
            x_bottleneck = layer_bn(x_bottleneck, encoder_out, encoder_compressed, cos, sin, decoder_mask, None)

        # Interleave: position 0 uses full path, positions 1+ use bottleneck path
        x = torch.zeros_like(x)
        x[:, 0:1] = x_full[:, 0:1]
        x[:, 1:] = x_bottleneck[:, 1:]

        x = self.decoder_norm(x)
        logits = self._get_decoder_logits(x)
        return logits
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def encode(self, encoder_input_ids: torch.Tensor,
               encoder_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode input sequence.
        
        Args:
            encoder_input_ids: (B, L_enc) encoder token indices
            encoder_mask: (B, L_enc) optional padding mask
            
        Returns:
            encoder_out: (B, L_enc, D) encoder representations
        """
        x = self.token_embedding(encoder_input_ids)
        
        L_enc = encoder_input_ids.shape[1]
        cos, sin = self.rope(L_enc, x.device)
        
        for layer in self.encoder_layers:
            x = layer(x, cos, sin, encoder_mask)
        
        x = self.encoder_norm(x)
        return x

    def forward(
        self,
        encoder_input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        decoder_mask: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        redshift_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Full forward pass.

        Args:
            encoder_input_ids: (B, L_enc) encoder input
            decoder_input_ids: (B, L_dec) decoder input (teacher forced)
            encoder_mask: (B, L_enc) encoder padding mask
            decoder_mask: (B, L_dec) decoder padding mask
            targets: (B, L_dec) target tokens for loss
            redshift_weight: scalar multiplier on the position-0 (redshift)
                cross-entropy term relative to the position-1+ (spectrum)
                term. The two terms are first reduced to per-token means,
                then combined as `redshift_weight * loss_redshift + loss_spectrum`.
                Default 1.0 keeps the two contributions on equal per-token
                footing. Set higher (e.g. 50) to force the model to learn
                the redshift token despite spectrum tokens dominating the
                position count.

        Returns:
            logits: (B, L_dec, vocab_size=2056)
            loss: scalar cross-entropy loss (if targets provided)
        """
        assert encoder_input_ids.shape[1] <= self.max_seq_len
        assert decoder_input_ids.shape[1] <= self.max_seq_len

        # Encode
        encoder_out = self.encode(encoder_input_ids, encoder_mask)

        # Decode
        logits = self.decode(decoder_input_ids, encoder_out, decoder_mask, encoder_mask)

        loss = None
        if targets is not None:
            B, L = targets.shape
            per_token = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view(B, L)
            valid = (targets != -100).float()
            n_red = valid[:, 0].sum().clamp(min=1.0)
            if L > 1:
                n_spec = valid[:, 1:].sum().clamp(min=1.0)
                loss_red = (per_token[:, 0] * valid[:, 0]).sum() / n_red
                loss_spec = (per_token[:, 1:] * valid[:, 1:]).sum() / n_spec
                loss = redshift_weight * loss_red + loss_spec
            else:
                loss = (per_token[:, 0] * valid[:, 0]).sum() / n_red

        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        encoder_input_ids: torch.Tensor,
        decoder_start_token: int = SOS_TOKEN,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """Autoregressive generation.
        
        Args:
            encoder_input_ids: (B, L_enc) encoder input
            decoder_start_token: Starting token for decoder
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            
        Returns:
            generated: (B, 1 + max_new_tokens) decoder output
        """
        self.eval()
        
        # Encode once
        encoder_out = self.encode(encoder_input_ids)
        
        # Start decoder with SOS token
        B = encoder_input_ids.shape[0]
        decoder_input_ids = torch.full((B, 1), decoder_start_token,
                                       dtype=torch.long, device=encoder_input_ids.device)
        
        for _ in range(max_new_tokens):
            # Decode
            logits = self.decode(decoder_input_ids, encoder_out)
            
            # Get logits for last position
            logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-p filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample from decoder vocabulary
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)

            # Stop if all EOS
            if (next_token == EOS_TOKEN).all():
                break

        return decoder_input_ids

    def ar_loss(
        self,
        encoder_input_ids: torch.Tensor,
        targets: torch.Tensor,
        max_generate_tokens: int = 50,
        redshift_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute autoregressive loss (full AR, no teacher forcing).

        For each position t, the model predicts tokens[0..t] given
        encoder_output and the model's own previous predictions (no teacher forcing).

        This is expensive but gives an honest training metric that matches
        the AR eval behavior.

        Args:
            encoder_input_ids: (B, L_enc) encoder input
            targets: (B, L_dec) target tokens [redshift, s1, s2, ..., sN, EOS]
            max_generate_tokens: truncate generation at this many tokens (for speed)
            redshift_weight: weight on position-0 (redshift) loss

        Returns:
            logits: (B, L, decoder_vocab_size) full logits sequence for logging/metrics
            loss: scalar loss (AR loss only)
        """
        B = encoder_input_ids.shape[0]

        encoder_out = self.encode(encoder_input_ids)

        gen_tokens = [[SOS_TOKEN] for _ in range(B)]
        all_logits = []

        max_len = min(targets.shape[1], max_generate_tokens + 1)

        for pos in range(max_len - 1):
            decoder_ids = torch.tensor(gen_tokens, dtype=torch.long, device=encoder_input_ids.device)
            logits = self.decode(decoder_ids, encoder_out)
            all_logits.append(logits[:, -1:, :])

            next_token_logits = logits[:, -1:, :].argmax(dim=-1).squeeze(-1).tolist()
            for i in range(B):
                gen_tokens[i].append(next_token_logits[i])

        logits_stacked = torch.cat(all_logits, dim=1)

        T = min(logits_stacked.shape[1], targets.shape[1] - 1)
        logits_trunc = logits_stacked[:, :T, :]
        targets_trunc = targets[:, 1:T+1]

        per_token = F.cross_entropy(
            logits_trunc.reshape(-1, self.vocab_size),
            targets_trunc.reshape(-1),
            ignore_index=-100,
            reduction='none',
        ).view(B, T)
        valid = (targets_trunc != -100).float()
        n_red = valid[:, 0].sum().clamp(min=1.0)
        n_spec = valid[:, 1:].sum().clamp(min=1.0) if T > 1 else torch.tensor(1.0, device=valid.device)
        loss_red = (per_token[:, 0] * valid[:, 0]).sum() / n_red
        loss_spec = (per_token[:, 1:] * valid[:, 1:]).sum() / n_spec if T > 1 else torch.tensor(0.0, device=per_token.device)
        loss = redshift_weight * loss_red + loss_spec

        logits_full = torch.zeros(B, targets.shape[1], self.vocab_size, device=logits_stacked.device)
        logits_full[:, :logits_stacked.shape[1], :] = logits_stacked

        return logits_full, loss


# Token encoding/decoding helpers (unchanged)
def encode_spectrum_token(lfq_index: torch.Tensor) -> torch.Tensor:
    return lfq_index + SPECTRUM_TOKEN_OFFSET

def decode_spectrum_token(token_id: torch.Tensor) -> torch.Tensor:
    return token_id - SPECTRUM_TOKEN_OFFSET

def encode_redshift_token(fsq_index: torch.Tensor) -> torch.Tensor:
    return fsq_index + REDSHIFT_TOKEN_OFFSET

def decode_redshift_token(token_id: torch.Tensor) -> torch.Tensor:
    return token_id - REDSHIFT_TOKEN_OFFSET

def is_spectrum_token(token_id: torch.Tensor) -> torch.Tensor:
    return (token_id >= SPECTRUM_TOKEN_OFFSET) & (token_id < REDSHIFT_TOKEN_OFFSET)

def is_redshift_token(token_id: torch.Tensor) -> torch.Tensor:
    return (token_id >= REDSHIFT_TOKEN_OFFSET) & (token_id < TOTAL_VOCAB_SIZE)


def build_encoder_input(
    redshift_token: Optional[torch.Tensor],
    spectrum_tokens: torch.Tensor,
    include_redshift: bool = True,
) -> torch.Tensor:
    """Build encoder input sequence.
    
    Args:
        redshift_token: Redshift token ID (or None)
        spectrum_tokens: (L,) spectrum token IDs
        include_redshift: If False, omit redshift (for Approach B)
        
    Returns:
        encoder_input: (L+2 or L+1,) sequence
    """
    seq = [SOS_TOKEN]
    
    if include_redshift and redshift_token is not None:
        seq.append(redshift_token.item() if isinstance(redshift_token, torch.Tensor) else redshift_token)
    
    seq.extend(spectrum_tokens.tolist() if isinstance(spectrum_tokens, torch.Tensor) else spectrum_tokens)
    seq.append(EOS_TOKEN)
    
    return torch.tensor(seq, dtype=torch.long)


def build_decoder_inputs(
    redshift_token: torch.Tensor,
    spectrum_tokens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build decoder input and target sequences for teacher forcing.
    
    Args:
        redshift_token: Redshift token ID
        spectrum_tokens: (L,) spectrum token IDs
        
    Returns:
        decoder_input: (L+2,) shifted right by 1: [SOS, redshift, s1, ..., sN]
        target: (L+2,) original: [redshift, s1, ..., sN, EOS]
    """
    redshift_val = redshift_token.item() if isinstance(redshift_token, torch.Tensor) else redshift_token
    spectrum_list = spectrum_tokens.tolist() if isinstance(spectrum_tokens, torch.Tensor) else spectrum_tokens
    
    decoder_input = [SOS_TOKEN, redshift_val] + spectrum_list
    target = [redshift_val] + spectrum_list + [EOS_TOKEN]
    
    return (
        torch.tensor(decoder_input, dtype=torch.long),
        torch.tensor(target, dtype=torch.long),
    )


def build_approach_a_sequences(
    redshift_token: torch.Tensor,
    spectrum_tokens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build encoder-decoder sequences for Approach A (joint redshift).
    
    Encoder sees: [SOS, redshift, s1, s2, ..., sN, EOS]
    Decoder input: [SOS, redshift, s1, s2, ..., sN]
    Target: [redshift, s1, s2, ..., sN, EOS]
    
    Returns:
        encoder_input, decoder_input, target
    """
    encoder_input = build_encoder_input(redshift_token, spectrum_tokens, include_redshift=True)
    decoder_input, target = build_decoder_inputs(redshift_token, spectrum_tokens)
    return encoder_input, decoder_input, target


def build_approach_b_sequences(
    redshift_token: torch.Tensor,
    spectrum_tokens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build encoder-decoder sequences for Approach B (mask redshift).
    
    Encoder sees: [SOS, s1, s2, ..., sN, EOS] (NO redshift!)
    Decoder input: [SOS, redshift, s1, s2, ..., sN] (teacher forced)
    Target: [redshift, s1, s2, ..., sN, EOS]
    
    The encoder has zero redshift information. The decoder must predict
    redshift at position 0 from cross-attention to the encoder's
    spectrum-only representation.
    
    Returns:
        encoder_input, decoder_input, target
    """
    encoder_input = build_encoder_input(redshift_token, spectrum_tokens, include_redshift=False)
    decoder_input, target = build_decoder_inputs(redshift_token, spectrum_tokens)
    return encoder_input, decoder_input, target
