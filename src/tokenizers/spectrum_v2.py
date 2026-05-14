"""
Spectrum Tokenizer V2
=====================
ConvNeXt-V2 based autoencoder with LFQ quantization for DESI spectra.

Improvements over V1:
- Top-hat 5-pixel smoothing preprocessing (smooths pixel noise before encoding)
- U-Net style skip connections from encoder to decoder at each scale
- Cross-attention decoder (decoder queries encoder skip connections at each scale)
- Codebook entropy loss (prevents codebook collapse)
- Lower commitment weight (0.05 vs V1's 0.25)

Architecture:
- Input: 2 channels (flux, ivar/istd) interpolated to fixed 8704-pixel grid
- TopHat: 5-pixel smoothing preprocessing
- Encoder: 4-stage ConvNeXt-V2 with progressive downsampling
  - Stem: 4x4 conv, stride 4 → 2176
  - Stage 1: 3 blocks (dim 96) → 544
  - Stage 2: downsample 2x + 3 blocks (dim 192) → 272
  - Stage 3: downsample 2x + 9 blocks (dim 384) → 136
  - Stage 4: downsample 2x + 3 blocks (dim 512) → 68
- Decoder: 4-stage U-Net with cross-attention
  - Stage 1: cross-attn(skip4) → 3 blocks → upsample 2x → 192
  - Stage 2: cross-attn(skip3) → 3 blocks → upsample 2x → 96
  - Stage 3: cross-attn(skip2) → 9 blocks → upsample 2x → 96
  - Stage 4: cross-attn(skip1) → 3 blocks → upsample 4x → 2 channels
- Quantizer: LFQ (dim=10, codebook=1024)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


LATENT_GRID_SIZE = 8704
N_TOKENS = 272


class LayerNorm1d(nn.Module):
    """LayerNorm for 1D conv features (B, C, L)."""

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
        return x


class ConvNeXtBlock1D(nn.Module):
    """ConvNeXt V2 block for 1D sequences."""

    def __init__(self, dim, kernel_size=7, mlp_ratio=4.0, drop_path=0.0):
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim
        )
        self.norm = LayerNorm1d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.pwconv1 = nn.Conv1d(dim, mlp_hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(mlp_hidden_dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input_x = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = input_x + self.drop_path(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class DownsampleBlock(nn.Module):
    """Downsampling block: LayerNorm -> Conv1d(stride=2)."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.norm = LayerNorm1d(in_dim)
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        return x


class UpsampleBlock(nn.Module):
    """Upsampling block: ConvTranspose1d(stride=2) -> LayerNorm."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_dim, out_dim, kernel_size=2, stride=2)
        self.norm = LayerNorm1d(out_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class CrossAttention(nn.Module):
    """Cross-attention: decoder features query encoder skip connection."""

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Conv1d(dim, dim, kernel_size=1)
        self.k_proj = nn.Conv1d(dim, dim, kernel_size=1)
        self.v_proj = nn.Conv1d(dim, dim, kernel_size=1)
        self.out_proj = nn.Conv1d(dim, dim, kernel_size=1)

    def forward(self, query, key_value):
        B, C, L_q = query.shape
        _, _, L_k = key_value.shape
        q = self.q_proj(query).reshape(B, self.num_heads, self.head_dim, L_q)
        k = self.k_proj(key_value).reshape(B, self.num_heads, self.head_dim, L_k)
        v = self.v_proj(key_value).reshape(B, self.num_heads, self.head_dim, L_k)
        attn = torch.einsum("bhdl,bhdk->bhlk", q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhlk,bhdk->bhdl", attn, v)
        out = out.reshape(B, C, L_q)
        return self.out_proj(out)


class EntropyLoss(nn.Module):
    """Codebook entropy loss — penalizes peaked code distributions."""

    def __init__(self, codebook_size: int, entropy_weight: float = 0.1):
        super().__init__()
        self.codebook_size = codebook_size
        self.entropy_weight = entropy_weight

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        codebook_counts = torch.bincount(
            indices.flatten(), minlength=self.codebook_size
        ).float()
        code_probs = codebook_counts / indices.numel()
        code_probs = code_probs.clamp(min=1e-10)
        entropy = -(code_probs * torch.log(code_probs)).sum()
        uniform_entropy = math.log(self.codebook_size)
        entropy_loss = (uniform_entropy - entropy) / uniform_entropy
        return entropy_loss * self.entropy_weight


class TopHatSmoothing(nn.Module):
    """Top-hat 5-pixel smoothing as preprocessing before encoder."""

    def __init__(self, channels=2):
        super().__init__()
        kernel = torch.ones(channels, 1, 5) / 5.0
        self.register_buffer("kernel", kernel)
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(x, self.kernel, padding=2, groups=self.channels)


class LookUpFreeQuantizerV2(nn.Module):
    """Look-up-Free Quantizer with entropy loss support."""

    def __init__(self, dim=10, codebook_size=1024, commitment_weight=0.05,
                 entropy_weight=0.1):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.project_in = nn.Conv1d(dim, dim, kernel_size=1)
        self.entropy_loss_fn = EntropyLoss(codebook_size, entropy_weight)

    def _compute_indices(self, z_q):
        bits = ((z_q + 1) / 2).long()
        powers = torch.arange(self.dim, device=z_q.device).view(1, -1, 1)
        return (bits * (2 ** powers)).sum(dim=1)

    def forward(self, z):
        z = self.project_in(z)
        z_q = torch.sign(z)
        z_q = z + (z_q - z).detach()

        commitment_loss = (
            F.mse_loss(z, torch.sign(z).detach()) * self.commitment_weight
        )
        entropy_loss = self.entropy_loss_fn(self._compute_indices(z_q))

        indices = self._compute_indices(z_q)

        return z_q, commitment_loss, entropy_loss, indices

    def encode(self, z):
        z = self.project_in(z)
        z_q = torch.sign(z)
        return self._compute_indices(z_q)

    def decode(self, indices):
        B, L = indices.shape
        z = torch.zeros(B, self.dim, L, device=indices.device)
        for i in range(self.dim):
            z[:, i, :] = ((indices // (2 ** i)) % 2).float() * 2 - 1
        return z


class SpectrumTokenizerV2(nn.Module):
    """Spectrum tokenizer V2: ConvNeXt-V2 autoencoder + LFQ + U-Net + cross-attention."""

    def __init__(
        self,
        in_channels=2,
        embedding_dim=10,
        codebook_size=1024,
        commitment_weight=0.05,
        entropy_weight=0.1,
        encoder_depths=(3, 3, 9, 3),
        encoder_dims=(96, 192, 384, 512),
        use_tophat=True,
        use_skip_connections=True,
        use_cross_attention=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size
        self.use_tophat = use_tophat
        self.use_skip_connections = use_skip_connections
        self.use_cross_attention = use_cross_attention

        self.tophat = TopHatSmoothing(channels=in_channels) if use_tophat else nn.Identity()

        # === ENCODER ===
        self.encoder_stem = nn.Sequential(
            nn.Conv1d(in_channels, encoder_dims[0], kernel_size=4, stride=4, padding=0),
            LayerNorm1d(encoder_dims[0]),
        )

        self.encoder_stages = nn.ModuleList()
        for i in range(len(encoder_depths)):
            stage = nn.ModuleList()
            if i > 0:
                stage.append(DownsampleBlock(encoder_dims[i - 1], encoder_dims[i]))
            for _ in range(encoder_depths[i]):
                stage.append(ConvNeXtBlock1D(encoder_dims[i]))
            self.encoder_stages.append(stage)

        self.pre_quant_norm = LayerNorm1d(encoder_dims[-1])
        self.quant_conv = nn.Conv1d(encoder_dims[-1], embedding_dim, kernel_size=1)

        self.quantizer = LookUpFreeQuantizerV2(
            dim=embedding_dim,
            codebook_size=codebook_size,
            commitment_weight=commitment_weight,
            entropy_weight=entropy_weight,
        )

        self.post_quant_conv = nn.Conv1d(embedding_dim, encoder_dims[-1], kernel_size=1)

        # === DECODER with U-Net skips ===
        # Decoder channel dims mirror encoder but in reverse, with last stage 96 for head
        decoder_dims = (encoder_dims[-1], encoder_dims[-2], encoder_dims[-3], 96)
        self.decoder_dims = decoder_dims

        # Skip projection layers to match decoder channel dims
        if use_skip_connections:
            self.skip_proj = nn.ModuleList([
                nn.Conv1d(encoder_dims[3], decoder_dims[0], kernel_size=1),  # skip4 -> dec1
                nn.Conv1d(encoder_dims[2], decoder_dims[1], kernel_size=1),  # skip3 -> dec2
                nn.Conv1d(encoder_dims[1], decoder_dims[2], kernel_size=1),  # skip2 -> dec3
                nn.Conv1d(encoder_dims[0], decoder_dims[3], kernel_size=1),  # skip1 -> dec4
            ])

        if use_cross_attention:
            self.cross_attn = nn.ModuleList([
                CrossAttention(decoder_dims[0]),
                CrossAttention(decoder_dims[1]),
                CrossAttention(decoder_dims[2]),
                CrossAttention(decoder_dims[3]),
            ])

        self.decoder_stages = nn.ModuleList()
        for i in range(len(decoder_dims)):
            stage = nn.ModuleList()
            for _ in range(encoder_depths[i]):
                stage.append(ConvNeXtBlock1D(decoder_dims[i]))
            if i < len(decoder_dims) - 1:
                stage.append(UpsampleBlock(decoder_dims[i], decoder_dims[i + 1]))
            self.decoder_stages.append(stage)

        self.decoder_head = nn.Sequential(
            nn.ConvTranspose1d(decoder_dims[-1], decoder_dims[-1], kernel_size=4, stride=4),
            LayerNorm1d(decoder_dims[-1]),
            nn.Conv1d(decoder_dims[-1], in_channels, kernel_size=1),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_to_grid(self, x, target_length=LATENT_GRID_SIZE):
        if x.shape[-1] != target_length:
            x = F.interpolate(x, size=target_length, mode="linear", align_corners=False)
        return x

    def normalize(self, x):
        flux = x[:, 0:1, :]
        istd = x[:, 1:2, :]
        positive_mask = flux > 0
        norm = (flux * positive_mask.float()).sum(dim=-1) / (
            positive_mask.sum(dim=-1).float() + 1.0
        )
        norm = torch.clamp(norm, min=0.1)
        norm_log = torch.log10(norm + 1.0)
        denorm = torch.clamp(10**norm_log - 1.0, min=0.1)
        flux_norm = (flux / denorm.unsqueeze(-1) - 1.0) * 0.2
        istd_norm = (istd / denorm.unsqueeze(-1)) * 0.2
        x_norm = torch.arcsinh(torch.cat([flux_norm, istd_norm], dim=1))
        return x_norm, denorm

    def denormalize(self, x_norm, denorm):
        x = torch.sinh(x_norm)
        flux_norm = x[:, 0:1, :]
        istd_norm = x[:, 1:2, :]
        flux = (flux_norm / 0.2 + 1.0) * denorm.unsqueeze(-1)
        istd = (istd_norm / 0.2) * denorm.unsqueeze(-1)
        return torch.cat([flux, istd], dim=1)

    def encode(self, x):
        x = self.interpolate_to_grid(x)
        x_norm, denorm = self.normalize(x)
        x = self.tophat(x_norm)
        x = self.encoder_stem(x)

        skips = []
        for stage in self.encoder_stages:
            for block in stage:
                x = block(x)
            skips.append(x)

        x = self.pre_quant_norm(x)
        x = self.quant_conv(x)
        indices = self.quantizer.encode(x)
        return indices, denorm, skips

    def decode(self, indices, denorm, skips):
        x = self.quantizer.decode(indices)
        x = self.post_quant_conv(x)

        for i, stage in enumerate(self.decoder_stages):
            if self.use_cross_attention:
                skip_proj = self.skip_proj[i] if self.use_skip_connections else None
                skip_for_attn = skips[-(i + 1)]
                if skip_for_attn.shape[-1] != x.shape[-1]:
                    skip_interp = F.interpolate(
                        skip_for_attn, size=x.shape[-1],
                        mode="linear", align_corners=False
                    )
                else:
                    skip_interp = skip_for_attn
                if self.use_skip_connections and skip_proj is not None:
                    skip_proj = skip_proj(skip_interp)
                    x = x + skip_proj
                x = x + self.cross_attn[i](x, skip_interp)
            elif self.use_skip_connections:
                skip = skips[-(i + 1)]
                target_len = skip.shape[-1]
                if x.shape[-1] != target_len:
                    x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
                skip_proj = self.skip_proj[i](skip)
                x = x + skip_proj

            for block in stage:
                x = block(x)

        x = self.decoder_head(x)
        x = self.denormalize(x, denorm)
        return x

    def forward(self, x):
        x_grid = self.interpolate_to_grid(x)
        x_norm, denorm = self.normalize(x_grid)

        x = self.tophat(x_norm)

        h = self.encoder_stem(x)
        skips = []
        for stage in self.encoder_stages:
            for block in stage:
                h = block(h)
            skips.append(h)

        h = self.pre_quant_norm(h)
        h = self.quant_conv(h)

        h_q, commit_loss, entropy_loss, indices = self.quantizer(h)

        h = self.post_quant_conv(h_q)

        for i, stage in enumerate(self.decoder_stages):
            if self.use_cross_attention:
                skip_interp = skips[-(i + 1)]
                if skip_interp.shape[-1] != h.shape[-1]:
                    skip_interp = F.interpolate(
                        skip_interp, size=h.shape[-1],
                        mode="linear", align_corners=False
                    )
                skip_proj = self.skip_proj[i](skip_interp) if self.use_skip_connections else None
                if skip_proj is not None:
                    h = h + skip_proj
                h = h + self.cross_attn[i](h, skip_interp)
            elif self.use_skip_connections:
                skip = skips[-(i + 1)]
                target_len = skip.shape[-1]
                if h.shape[-1] != target_len:
                    h = F.interpolate(h, size=target_len, mode="linear", align_corners=False)
                h = h + self.skip_proj[i](skip)

            for block in stage:
                h = block(h)

        recon_norm = self.decoder_head(h)
        recon = self.denormalize(recon_norm, denorm)

        recon_loss = F.mse_loss(recon, x_grid)
        quant_loss = commit_loss + entropy_loss

        loss = {
            "total": recon_loss + quant_loss,
            "recon": recon_loss,
            "quant": quant_loss,
            "commit": commit_loss,
            "entropy": entropy_loss,
        }

        return recon, loss, indices


def test_v2():
    """Quick shape test."""
    batch_size = 2
    seq_len = 7781

    model = SpectrumTokenizerV2(
        use_skip_connections=True,
        use_cross_attention=True,
    )
    x = torch.randn(batch_size, 2, seq_len)

    recon, loss, indices = model(x)
    print(f"Input shape:   {x.shape}")
    print(f"Output shape: {recon.shape}")
    print(f"Indices shape: {indices.shape}")
    print(f"N tokens:     {indices.shape[1]}")
    print(f"Losses:       { {k: f'{v.item():.4f}' for k, v in loss.items()} }")

    indices2, denorm, skips = model.encode(x)
    recon2 = model.decode(indices2, denorm, skips)
    print(f"\nEncode->Decode shape: {recon2.shape}")
    print(f"Max recon diff: {(recon - recon2).abs().max().item():.6f}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {n_params:,}")

    return model


if __name__ == "__main__":
    test_v2()