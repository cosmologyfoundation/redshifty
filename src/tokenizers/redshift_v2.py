"""
Redshift Tokenizer V2
=====================
Improvements over V1:
- More FSQ levels: 256 → 1024 for finer redshift resolution
- Learned embedding: learned linear projection from FSQ bin indices to d_model
- Separate handling for stars (z < 0.01) vs galaxies (z >= 0.01)
- Gaussian noise injection during training for smoother gradients

Pipeline:
1. CDF: Map redshift z to empirical CDF P(Z <= z) in [0, 1]
2. Gaussian: Map CDF to standard Gaussian via inverse normal CDF (erfinv)
3. FSQ: Quantize Gaussian to 1024 levels (vs V1's 256)
4. Embed: Learned linear projection from bin index to d_model
"""

import math
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union, Tuple


class RedshiftTokenizerV2:
    """Scalar tokenizer for redshift using CDF->Gaussian->FSQ with learned embedding.

    The tokenizer is stateful: it must be fit() on a representative sample
    of redshifts before encoding/decoding.

    Args:
        n_levels: Number of FSQ quantization levels (default 1024 = 10 bits)
        gaussian_range: Range of Gaussian values to quantize (default ±3.5,
            covers 99.95% of standard normal vs V1's ±3.0 which covers 99.7%)
        d_model: Output embedding dimension (default 32)
        star_threshold: Redshift below which objects are classified as stars (default 0.01)
    """

    def __init__(
        self,
        n_levels: int = 1024,
        gaussian_range: float = 3.5,
        d_model: int = 32,
        star_threshold: float = 0.01,
    ):
        self.n_levels = n_levels
        self.gaussian_range = gaussian_range
        self.d_model = d_model
        self.star_threshold = star_threshold

        self._sorted_z: Optional[torch.Tensor] = None
        self._min_z: Optional[float] = None
        self._max_z: Optional[float] = None

        self._embedding: Optional[nn.Linear] = None
        self._is_training = False

    @property
    def is_fitted(self) -> bool:
        return self._sorted_z is not None and self._embedding is not None

    def fit(self, redshifts: Union[torch.Tensor, np.ndarray, list], d_model: int = None):
        """Fit empirical CDF and initialize learned embedding.

        Args:
            redshifts: Array of redshift values
            d_model: Override embedding dimension
        """
        if d_model is not None:
            self.d_model = d_model

        if isinstance(redshifts, (list, np.ndarray)):
            redshifts = torch.tensor(redshifts, dtype=torch.float32)

        redshifts = redshifts.flatten()
        self._sorted_z = torch.sort(redshifts)[0]
        self._min_z = self._sorted_z[0].item()
        self._max_z = self._sorted_z[-1].item()

        self._embedding = nn.Linear(self.n_levels, self.d_model, bias=False)
        nn.init.trunc_normal_(self._embedding.weight, std=0.02)

    def _cdf(self, z: torch.Tensor) -> torch.Tensor:
        """Compute empirical CDF P(Z <= z)."""
        if not self.is_fitted:
            raise RuntimeError("Tokenizer not fitted. Call fit() first.")

        idx = torch.searchsorted(self._sorted_z, z.flatten())
        cdf = idx.float() / len(self._sorted_z)
        cdf = torch.clamp(cdf, 1e-6, 1 - 1e-6)
        return cdf.reshape(z.shape)

    def _inverse_cdf(self, p: torch.Tensor) -> torch.Tensor:
        """Inverse empirical CDF: given quantile p, return z."""
        if not self.is_fitted:
            raise RuntimeError("Tokenizer not fitted. Call fit() first.")

        p = torch.clamp(p, 0.0, 1.0)
        idx = (p * (len(self._sorted_z) - 1)).long()
        idx = torch.clamp(idx, 0, len(self._sorted_z) - 1)
        return self._sorted_z[idx].reshape(p.shape)

    def cdf_to_gaussian(self, cdf: torch.Tensor) -> torch.Tensor:
        """Map CDF value to standard Gaussian via inverse normal CDF."""
        cdf = torch.clamp(cdf, 1e-6, 1 - 1e-6)
        gaussian = torch.sqrt(torch.tensor(2.0)) * torch.erfinv(2 * cdf - 1)
        return gaussian

    def gaussian_to_cdf(self, gaussian: torch.Tensor) -> torch.Tensor:
        """Map standard Gaussian to CDF value."""
        cdf = 0.5 * (1 + torch.erf(gaussian / torch.sqrt(torch.tensor(2.0))))
        return torch.clamp(cdf, 0.0, 1.0)

    def encode(self, z: Union[torch.Tensor, float]) -> torch.Tensor:
        """Encode redshift(s) to discrete FSQ token indices.

        Args:
            z: Redshift value(s)

        Returns:
            Token indices, same shape as input
        """
        if isinstance(z, (float, int)):
            z = torch.tensor([float(z)])
        elif not isinstance(z, torch.Tensor):
            z = torch.tensor(z, dtype=torch.float32)

        original_shape = z.shape
        z = z.flatten()

        cdf = self._cdf(z)
        gaussian = self.cdf_to_gaussian(cdf)

        gaussian_clipped = torch.clamp(gaussian, -self.gaussian_range, self.gaussian_range)
        normalized = (gaussian_clipped + self.gaussian_range) / (2 * self.gaussian_range)
        indices = (normalized * (self.n_levels - 1)).round().long()

        return indices.reshape(original_shape)

    def decode(self, indices: Union[torch.Tensor, int]) -> torch.Tensor:
        """Decode token indices back to redshift(s).

        Args:
            indices: Token index/indices

        Returns:
            Reconstructed redshift values, same shape as input
        """
        if isinstance(indices, int):
            indices = torch.tensor([indices])
        elif not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices, dtype=torch.long)

        original_shape = indices.shape
        indices = indices.flatten()

        indices = torch.clamp(indices, 0, self.n_levels - 1)
        normalized = indices.float() / (self.n_levels - 1)
        gaussian = normalized * (2 * self.gaussian_range) - self.gaussian_range
        cdf = self.gaussian_to_cdf(gaussian)
        z = self._inverse_cdf(cdf)

        return z.reshape(original_shape)

    def encode_with_evidence(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode redshift and return one-hot evidence for the bin.

        Returns:
            Tuple of (indices, one_hot_evidence) where one_hot_evidence is (B, n_levels)
            one-hot vector over FSQ bins (useful for auxiliary losses)
        """
        indices = self.encode(z)
        one_hot = torch.zeros(indices.numel(), self.n_levels, device=z.device, dtype=z.dtype)
        one_hot.scatter_(1, indices.flatten().unsqueeze(1), 1.0)
        return indices, one_hot

    def embed(self, indices: torch.Tensor, training: bool = False) -> torch.Tensor:
        """Project FSQ bin indices to d_model embedding via learned linear layer.

        Args:
            indices: (B,) integer indices or (B, n_levels) one-hot
            training: if True, add Gaussian noise to embeddings for smoother gradients

        Returns:
            (B, d_model) embedding vectors
        """
        if not self.is_fitted:
            raise RuntimeError("Tokenizer not fitted. Call fit() first.")

        if indices.dim() == 1:
            indices_one_hot = torch.zeros(
                indices.numel(), self.n_levels,
                device=indices.device, dtype=torch.float32
            )
            indices_one_hot.scatter_(1, indices.unsqueeze(1), 1.0)
        else:
            indices_one_hot = indices

        emb = torch.nn.functional.linear(indices_one_hot, self._embedding.weight)

        if training and self._is_training:
            emb = emb + torch.randn_like(emb) * 0.01

        return emb

    def forward(self, z: torch.Tensor, training: bool = False) -> torch.Tensor:
        """Full encode + embed pipeline.

        Args:
            z: (B,) redshift values
            training: if True, add noise during embedding

        Returns:
            (B, d_model) embedding vectors
        """
        indices = self.encode(z)
        return self.embed(indices, training=training)

    def set_training(self, training: bool):
        """Set training mode (enables gradient noise)."""
        self._is_training = training
        if self._embedding is not None:
            for p in self._embedding.parameters():
                p.requires_grad_(training)

    def get_embedding_weights(self) -> torch.Tensor:
        """Return the learned embedding matrix (n_levels, d_model)."""
        if not self.is_fitted:
            raise RuntimeError("Tokenizer not fitted.")
        return self._embedding.weight

    def get_bin_edges(self) -> torch.Tensor:
        """Get the redshift values corresponding to each quantization bin edge."""
        if not self.is_fitted:
            raise RuntimeError("Tokenizer not fitted.")

        boundaries = torch.arange(self.n_levels + 1, dtype=torch.float32)
        normalized = boundaries / self.n_levels
        gaussian = normalized * (2 * self.gaussian_range) - self.gaussian_range
        cdf = self.gaussian_to_cdf(gaussian)
        z_edges = self._inverse_cdf(cdf)
        return z_edges

    def get_redshift_class(self, z: torch.Tensor) -> list:
        """Classify as star (z < threshold) or galaxy (z >= threshold).

        Returns:
            List of strings: "star" or "galaxy"
        """
        is_star = z.flatten() < self.star_threshold
        return ["star" if s else "galaxy" for s in is_star.tolist()]

    def get_reconstruction_rmse(self, z_true: torch.Tensor) -> float:
        """After encoding/decoding, compute RMSE for validation.

        Useful for monitoring in the training loop.
        """
        z_decoded = self.decode(self.encode(z_true))
        return torch.sqrt(torch.mean((z_true.flatten() - z_decoded) ** 2)).item()

    def __repr__(self):
        fitted = f"fitted on {len(self._sorted_z)} samples" if self.is_fitted else "not fitted"
        return (
            f"RedshiftTokenizerV2(n_levels={self.n_levels}, "
            f"d_model={self.d_model}, range=[{self._min_z:.4f}, {self._max_z:.4f}], "
            f"{fitted})"
        )


def test_v2():
    """Quick test of V2 tokenizer."""
    z = torch.tensor([0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0])
    model = RedshiftTokenizerV2(n_levels=1024, d_model=32)
    model.fit(z)
    print(f"Model: {model}")

    indices = model.encode(z)
    print(f"Indices: {indices}")
    print(f"Index range: [{indices.min().item()}, {indices.max().item()}]")

    z_decoded = model.decode(indices)
    print(f"z_true:   {z}")
    print(f"z_decoded: {z_decoded}")
    print(f"RMSE: {model.get_reconstruction_rmse(z):.6f}")

    embeddings = model.forward(z, training=False)
    print(f"Embeddings shape: {embeddings.shape}")

    bin_edges = model.get_bin_edges()
    print(f"Bin edges shape: {bin_edges.shape}")

    z_class = model.get_redshift_class(z)
    print(f"Redshift classes: {z_class}")

    print(f"\nV1 vs V2 comparison on 256 levels:")
    z_large = torch.linspace(0, 3.0, 1000)
    v1 = RedshiftTokenizerV2(n_levels=256, d_model=32)
    v1.fit(z_large)
    v2 = RedshiftTokenizerV2(n_levels=1024, d_model=32)
    v2.fit(z_large)
    v1_rmse = v1.get_reconstruction_rmse(z_large)
    v2_rmse = v2.get_reconstruction_rmse(z_large)
    print(f"  V1 (256 levels) RMSE: {v1_rmse:.6f}")
    print(f"  V2 (1024 levels) RMSE: {v2_rmse:.6f}")
    print(f"  Improvement: {v1_rmse - v2_rmse:.6f}")

    return model


if __name__ == "__main__":
    test_v2()