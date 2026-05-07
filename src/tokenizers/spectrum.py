"""
Spectrum Tokenizer
==================
ConvNeXt-V2 based autoencoder with LFQ quantization for DESI spectra.

Architecture (based on AION paper):
- Input: 2 channels (flux, ivar/istd) interpolated to fixed 8704-pixel grid
- Encoder: 4-stage ConvNeXt-V2 with progressive downsampling
  - Stem: 4x4 conv, stride 4 → 2176
  - Stage 1: 3 ConvNeXt blocks (dim 96)
  - Stage 2: downsample 2x + 3 blocks (dim 192) → 1088
  - Stage 3: downsample 2x + 9 blocks (dim 384) → 544
  - Stage 4: downsample 2x + 3 blocks (dim 512) → 272
- Latent: 272 tokens (we pad to 273)
- Quantizer: Look-up-Free Quantizer (LFQ) with codebook size 1024
- Decoder: mirror of encoder with upsampling
- Output: reconstructed flux (+ optional mask)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


# Fixed latent grid (matches AION paper)
LATENT_GRID_SIZE = 8704
N_TOKENS = 273


class LayerNorm1d(nn.Module):
    """LayerNorm for 1D conv features (B, C, L)."""
    
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
    
    def forward(self, x):
        # x: (B, C, L)
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
        return x


class ConvNeXtBlock1D(nn.Module):
    """ConvNeXt V2 block for 1D sequences.
    
    Depthwise conv -> LayerNorm -> pointwise (expand) -> GELU -> pointwise (project) -> residual
    """
    
    def __init__(self, dim, kernel_size=7, mlp_ratio=4.0, drop_path=0.0):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
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


class LookUpFreeQuantizer(nn.Module):
    """Look-up-Free Quantizer (LFQ) - binary quantization.
    
    Each dimension is quantized to {-1, +1}, giving exactly 2^dim codes.
    With dim=10, codebook_size = 2^10 = 1024.
    
    Uses straight-through estimator with commitment + entropy losses.
    """
    
    def __init__(self, dim=10, codebook_size=1024, commitment_weight=0.25, entropy_weight=0.1):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.entropy_weight = entropy_weight
        
        self.project_in = nn.Conv1d(dim, dim, kernel_size=1)
        
        # Temperature for controlling sharpness of sign
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(self, z):
        """Quantize latents.
        
        Args:
            z: (B, dim, L) continuous latent
            
        Returns:
            z_q: (B, dim, L) quantized latent
            loss: quantization loss
            indices: (B, L) token indices
        """
        z = self.project_in(z)
        
        # Scale by temperature to control sharpness
        z_scaled = z / (self.temperature.abs() + 1e-8)
        
        # Binary quantization: sign(z) → {-1, +1}
        z_q = torch.sign(z_scaled)
        
        # Straight-through estimator
        z_q = z_scaled + (z_q - z_scaled).detach()
        
        # Commitment loss
        commit_loss = F.mse_loss(z, z_q.detach()) * self.commitment_weight
        
        # Entropy loss: encourage balanced code usage
        # Compute probability of +1 for each dimension
        p = torch.sigmoid(z_scaled)  # (B, dim, L)
        # Binary entropy: H = -p*log(p) - (1-p)*log(1-p)
        eps = 1e-8
        entropy = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))
        entropy_loss = -entropy.mean() * self.entropy_weight  # Negative to maximize entropy
        
        loss = commit_loss + entropy_loss
        
        # Compute indices: binary to integer
        # z_q is in {-1, 1}, map to {0, 1} then compute binary number
        bits = ((z_q + 1) / 2).long()  # (B, dim, L) with values {0, 1}
        # Compute binary code: sum(bit_i * 2^i) for i in [0, dim-1]
        powers = torch.arange(self.dim, device=z.device).view(1, -1, 1)
        indices = (bits * (2 ** powers)).sum(dim=1)  # (B, L)
        
        return z_q, loss, indices
    
    def encode(self, z):
        """Encode to discrete indices."""
        z = self.project_in(z)
        z_scaled = z / (self.temperature.abs() + 1e-8)
        z_q = torch.sign(z_scaled)
        
        bits = ((z_q + 1) / 2).long()
        powers = torch.arange(self.dim, device=z.device).view(1, -1, 1)
        indices = (bits * (2 ** powers)).sum(dim=1)
        return indices
    
    def decode(self, indices):
        """Decode from discrete indices."""
        # Convert integer to binary
        B, L = indices.shape
        z = torch.zeros(B, self.dim, L, device=indices.device)
        
        for i in range(self.dim):
            z[:, i, :] = ((indices // (2 ** i)) % 2).float() * 2 - 1
        
        return z


class SpectrumTokenizer(nn.Module):
    """Spectrum tokenizer: ConvNeXt-V2 autoencoder + LFQ quantization.
    
    Interpolates input to fixed 8704-pixel grid before encoding.
    
    Args:
        in_channels: Input channels (2 for flux+ivar)
        latent_channels: Latent dimension before quantization (default: 512)
        embedding_dim: Quantization dimension (default: 10)
        codebook_size: Number of discrete codes (default: 1024)
        encoder_depths: Number of ConvNeXt blocks per stage
        encoder_dims: Channel dimensions per stage
        decoder_depths: Number of ConvNeXt blocks per stage
        decoder_dims: Channel dimensions per stage
    """
    
    def __init__(
        self,
        in_channels=2,
        latent_channels=512,
        embedding_dim=10,
        codebook_size=1024,
        encoder_depths=(3, 3, 9, 3),
        encoder_dims=(96, 192, 384, 512),
        decoder_depths=(3, 3, 9, 3),
        decoder_dims=(384, 192, 96, 96),
        commitment_weight=0.25,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size
        
        # === ENCODER ===
        self.encoder_stem = nn.Sequential(
            nn.Conv1d(in_channels, encoder_dims[0], kernel_size=4, stride=4, padding=0),
            LayerNorm1d(encoder_dims[0]),
        )
        
        self.encoder_stages = nn.ModuleList()
        for i in range(len(encoder_depths)):
            stage = nn.ModuleList()
            # Downsampling (except first stage which uses stem)
            if i > 0:
                stage.append(DownsampleBlock(encoder_dims[i-1], encoder_dims[i]))
            # ConvNeXt blocks
            for _ in range(encoder_depths[i]):
                stage.append(ConvNeXtBlock1D(encoder_dims[i]))
            self.encoder_stages.append(stage)
        
        # Pre-quantization projection
        self.pre_quant_norm = LayerNorm1d(encoder_dims[-1])
        self.quant_conv = nn.Conv1d(encoder_dims[-1], embedding_dim, kernel_size=1)
        
        # Quantizer
        self.quantizer = LookUpFreeQuantizer(
            dim=embedding_dim,
            codebook_size=codebook_size,
            commitment_weight=commitment_weight,
        )
        
        # Post-quantization projection
        self.post_quant_conv = nn.Conv1d(embedding_dim, decoder_dims[0], kernel_size=1)
        
        # === DECODER ===
        self.decoder_stages = nn.ModuleList()
        for i in range(len(decoder_depths)):
            stage = nn.ModuleList()
            # ConvNeXt blocks
            for _ in range(decoder_depths[i]):
                stage.append(ConvNeXtBlock1D(decoder_dims[i]))
            # Upsampling (except last stage)
            if i < len(decoder_depths) - 1:
                stage.append(UpsampleBlock(decoder_dims[i], decoder_dims[i+1]))
            self.decoder_stages.append(stage)
        
        # Output head: upsample by 4 to match stem stride
        self.decoder_head = nn.Sequential(
            nn.ConvTranspose1d(decoder_dims[-1], decoder_dims[-1], kernel_size=4, stride=4),
            LayerNorm1d(decoder_dims[-1]),
            nn.Conv1d(decoder_dims[-1], in_channels, kernel_size=1),
        )
        
        # Initialize weights
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
        """Interpolate spectrum to fixed grid length."""
        if x.shape[-1] != target_length:
            x = F.interpolate(x, size=target_length, mode='linear', align_corners=False)
        return x
    
    def encode(self, x):
        """Encode spectrum to quantized tokens.
        
        Args:
            x: (B, in_channels, L) input spectrum
            
        Returns:
            indices: (B, dim, n_tokens) discrete token indices
            denorm: (B,) normalization factor for denormalization
        """
        x = self.interpolate_to_grid(x)
        x_norm, denorm = self.normalize(x)
        x = self.encoder_stem(x_norm)
        
        for stage in self.encoder_stages:
            for block in stage:
                x = block(x)
        
        x = self.pre_quant_norm(x)
        x = self.quant_conv(x)
        
        indices = self.quantizer.encode(x)
        
        return indices, denorm
    
    def decode(self, indices, denorm):
        """Decode tokens to spectrum.
        
        Args:
            indices: (B, dim, n_tokens) discrete token indices
            denorm: (B,) normalization factor
            
        Returns:
            x: (B, in_channels, LATENT_GRID_SIZE) reconstructed spectrum
        """
        x = self.quantizer.decode(indices)
        x = self.post_quant_conv(x)
        
        for stage in self.decoder_stages:
            for block in stage:
                x = block(x)
        
        x = self.decoder_head(x)
        
        # Denormalize
        x = self.denormalize(x, denorm)
        
        return x
    
    def normalize(self, x):
        """Normalize spectrum to zero-mean, unit-variance-like range.
        
        AION-style normalization:
        1. Compute robust median flux (mean of unmasked pixels, clamped)
        2. log10 compress the normalization factor
        3. Normalize flux: (flux / norm - 1) * input_scaling
        4. Normalize istd similarly
        5. Stack and apply arcsinh for additional range compression
        
        Args:
            x: (B, 2, L) where x[:,0] = flux, x[:,1] = istd
            
        Returns:
            x_norm: (B, 2, L) normalized spectrum
            norm_factor: (B,) normalization factor for denormalization
        """
        flux = x[:, 0:1, :]  # (B, 1, L)
        istd = x[:, 1:2, :]  # (B, 1, L)
        
        # Compute robust median (mean of positive flux)
        positive_mask = flux > 0
        norm = (flux * positive_mask.float()).sum(dim=-1) / (positive_mask.sum(dim=-1).float() + 1.0)
        norm = torch.clamp(norm, min=0.1)  # Avoid division by zero
        
        # log10 compression of normalization factor
        norm_log = torch.log10(norm + 1.0)
        
        # Denormalization factor
        denorm = torch.clamp(10 ** norm_log - 1.0, min=0.1)
        
        # Normalize flux and istd
        flux_norm = (flux / denorm.unsqueeze(-1) - 1.0) * 0.2
        istd_norm = (istd / denorm.unsqueeze(-1)) * 0.2
        
        # Stack and apply arcsinh for range compression
        x_norm = torch.arcsinh(torch.cat([flux_norm, istd_norm], dim=1))
        
        return x_norm, denorm
    
    def denormalize(self, x_norm, denorm):
        """Denormalize reconstructed spectrum.
        
        Args:
            x_norm: (B, 2, L) normalized spectrum (after inverse arcsinh)
            denorm: (B,) normalization factor
            
        Returns:
            x: (B, 2, L) denormalized spectrum
        """
        # Inverse arcsinh
        x = torch.sinh(x_norm)
        
        flux_norm = x[:, 0:1, :]
        istd_norm = x[:, 1:2, :]
        
        # Denormalize
        flux = (flux_norm / 0.2 + 1.0) * denorm.unsqueeze(-1)
        istd = (istd_norm / 0.2) * denorm.unsqueeze(-1)
        
        x = torch.cat([flux, istd], dim=1)
        
        return x
    
    def forward(self, x):
        """Full forward pass: encode + quantize + decode.
        
        Args:
            x: (B, in_channels, L) input spectrum
            
        Returns:
            recon: (B, in_channels, LATENT_GRID_SIZE) reconstructed spectrum
            loss: dict with quantization and reconstruction losses
            indices: (B, n_tokens) discrete token indices
        """
        # Interpolate to fixed grid
        x_grid = self.interpolate_to_grid(x)
        
        # Normalize
        x_norm, denorm = self.normalize(x_grid)
        
        # Encode
        h = self.encoder_stem(x_norm)
        for stage in self.encoder_stages:
            for block in stage:
                h = block(h)
        
        h = self.pre_quant_norm(h)
        h = self.quant_conv(h)
        
        # Quantize
        h_q, quant_loss, indices = self.quantizer(h)
        
        # Decode
        h = self.post_quant_conv(h_q)
        for stage in self.decoder_stages:
            for block in stage:
                h = block(h)
        
        recon_norm = self.decoder_head(h)
        
        # Denormalize
        recon = self.denormalize(recon_norm, denorm)
        
        # Reconstruction loss (on interpolated grid, denormalized)
        recon_loss = F.mse_loss(recon, x_grid)
        
        loss = {
            "total": recon_loss + quant_loss,
            "recon": recon_loss,
            "quant": quant_loss,
        }
        
        return recon, loss, indices


def test_tokenizer_shapes():
    """Quick shape test."""
    batch_size = 2
    seq_len = 7781  # Our DESI data length
    
    model = SpectrumTokenizer()
    x = torch.randn(batch_size, 2, seq_len)
    
    # Forward
    recon, loss, indices = model(x)
    print(f"Input shape:   {x.shape}")
    print(f"Output shape:  {recon.shape}")
    print(f"Indices shape: {indices.shape}")
    print(f"N tokens:      {indices.shape[1]}")
    print(f"Losses:        { {k: f'{v.item():.4f}' for k, v in loss.items()} }")
    
    # Encode/decode round-trip
    indices2, denorm = model.encode(x)
    recon2 = model.decode(indices2, denorm)
    print(f"\nEncode -> Decode shape: {recon2.shape}")
    print(f"Max reconstruction diff: {(recon - recon2).abs().max().item():.6f}")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {n_params:,}")
    
    return model


if __name__ == "__main__":
    test_tokenizer_shapes()
