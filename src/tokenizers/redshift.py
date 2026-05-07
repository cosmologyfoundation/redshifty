"""
Redshift Scalar Tokenizer
=========================
Tokenizes scalar redshift values using CDF -> Gaussian -> FSQ pipeline.

Pipeline:
1. CDF: Map redshift z to its empirical CDF value P(Z <= z) in [0, 1]
2. Gaussian: Map CDF to standard Gaussian via inverse normal CDF (erfinv)
3. FSQ: Quantize Gaussian to discrete levels (default 256)

This ensures uniform utilization of quantization bins regardless of the
original redshift distribution (which is heavily skewed toward z~0 for stars).
"""

import torch
import numpy as np
from typing import Optional, Union


class RedshiftTokenizer:
    """Scalar tokenizer for redshift using CDF->Gaussian->FSQ.
    
    The tokenizer is stateful: it must be fit() on a representative
    sample of redshifts before encoding/decoding.
    
    Args:
        n_levels: Number of quantization levels (default 256 = 8 bits)
        gaussian_range: Range of Gaussian values to quantize (default ±3.0,
            which covers 99.7% of standard normal distribution)
    """
    
    def __init__(self, n_levels: int = 256, gaussian_range: float = 3.0):
        self.n_levels = n_levels
        self.gaussian_range = gaussian_range
        self._sorted_z: Optional[torch.Tensor] = None
        self._min_z: Optional[float] = None
        self._max_z: Optional[float] = None
    
    @property
    def is_fitted(self) -> bool:
        return self._sorted_z is not None
    
    def fit(self, redshifts: Union[torch.Tensor, np.ndarray, list]):
        """Fit empirical CDF on training redshifts.
        
        Args:
            redshifts: Array of redshift values
        """
        if isinstance(redshifts, (list, np.ndarray)):
            redshifts = torch.tensor(redshifts, dtype=torch.float32)
        
        redshifts = redshifts.flatten()
        self._sorted_z = torch.sort(redshifts)[0]
        self._min_z = self._sorted_z[0].item()
        self._max_z = self._sorted_z[-1].item()
    
    def _cdf(self, z: torch.Tensor) -> torch.Tensor:
        """Compute empirical CDF P(Z <= z)."""
        if not self.is_fitted:
            raise RuntimeError("Tokenizer not fitted. Call fit() first.")
        
        # For each z, find proportion of sorted_redshifts <= z
        # searchsorted returns the index where z would be inserted
        idx = torch.searchsorted(self._sorted_z, z.flatten())
        # Normalize to [0, 1], with small epsilon to avoid exact 0/1
        cdf = idx.float() / len(self._sorted_z)
        cdf = torch.clamp(cdf, 1e-6, 1 - 1e-6)
        return cdf.reshape(z.shape)
    
    def _inverse_cdf(self, p: torch.Tensor) -> torch.Tensor:
        """Inverse empirical CDF: given quantile p, return z."""
        if not self.is_fitted:
            raise RuntimeError("Tokenizer not fitted. Call fit() first.")
        
        p = torch.clamp(p, 0.0, 1.0)
        # Map quantile to index in sorted array
        idx = (p * (len(self._sorted_z) - 1)).long()
        idx = torch.clamp(idx, 0, len(self._sorted_z) - 1)
        return self._sorted_z[idx].reshape(p.shape)
    
    def cdf_to_gaussian(self, cdf: torch.Tensor) -> torch.Tensor:
        """Map CDF value to standard Gaussian via inverse normal CDF.
        
        Φ^{-1}(p) = sqrt(2) * erfinv(2p - 1)
        """
        # Clamp to avoid infinities at exactly 0 or 1
        cdf = torch.clamp(cdf, 1e-6, 1 - 1e-6)
        gaussian = torch.sqrt(torch.tensor(2.0)) * torch.erfinv(2 * cdf - 1)
        return gaussian
    
    def gaussian_to_cdf(self, gaussian: torch.Tensor) -> torch.Tensor:
        """Map standard Gaussian to CDF value.
        
        Φ(g) = 0.5 * (1 + erf(g / sqrt(2)))
        """
        cdf = 0.5 * (1 + torch.erf(gaussian / torch.sqrt(torch.tensor(2.0))))
        return torch.clamp(cdf, 0.0, 1.0)
    
    def encode(self, z: Union[torch.Tensor, float]) -> torch.Tensor:
        """Encode redshift(s) to discrete token indices.
        
        Args:
            z: Redshift value(s), scalar or tensor
            
        Returns:
            Token indices, same shape as input
        """
        if isinstance(z, (float, int)):
            z = torch.tensor([float(z)])
        elif not isinstance(z, torch.Tensor):
            z = torch.tensor(z, dtype=torch.float32)
        
        original_shape = z.shape
        z = z.flatten()
        
        # Step 1: CDF transform
        cdf = self._cdf(z)
        
        # Step 2: Gaussian transform
        gaussian = self.cdf_to_gaussian(cdf)
        
        # Step 3: FSQ - quantize Gaussian to n_levels
        # Map from [-gaussian_range, +gaussian_range] to [0, n_levels-1]
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
        
        # Step 1: FSQ -> Gaussian
        indices = torch.clamp(indices, 0, self.n_levels - 1)
        normalized = indices.float() / (self.n_levels - 1)  # [0, 1]
        gaussian = normalized * (2 * self.gaussian_range) - self.gaussian_range
        
        # Step 2: Gaussian -> CDF
        cdf = self.gaussian_to_cdf(gaussian)
        
        # Step 3: CDF -> z
        z = self._inverse_cdf(cdf)
        
        return z.reshape(original_shape)
    
    def encode_batch(self, z: torch.Tensor) -> torch.Tensor:
        """Batch encode redshifts. Alias for encode()."""
        return self.encode(z)
    
    def decode_batch(self, indices: torch.Tensor) -> torch.Tensor:
        """Batch decode indices. Alias for decode()."""
        return self.decode(indices)
    
    def get_bin_edges(self) -> torch.Tensor:
        """Get the redshift values corresponding to each quantization bin edge.
        
        Returns:
            Tensor of shape (n_levels + 1,) with bin edge redshift values
        """
        if not self.is_fitted:
            raise RuntimeError("Tokenizer not fitted. Call fit() first.")
        
        # Get Gaussian values for each bin boundary (including edges)
        # We want n_levels + 1 boundaries: -inf, bin_edges..., +inf
        # Map to [-gaussian_range, +gaussian_range]
        boundaries = torch.arange(self.n_levels + 1, dtype=torch.float32)
        normalized = boundaries / self.n_levels  # [0, 1] with n_levels+1 points
        gaussian = normalized * (2 * self.gaussian_range) - self.gaussian_range
        
        # Convert to CDF
        cdf = self.gaussian_to_cdf(gaussian)
        
        # Convert to z
        z_edges = self._inverse_cdf(cdf)
        
        return z_edges
    
    def __repr__(self):
        fitted_str = f"fitted on {len(self._sorted_z)} samples" if self.is_fitted else "not fitted"
        return f"RedshiftTokenizer(n_levels={self.n_levels}, range=[{self._min_z:.4f}, {self._max_z:.4f}] if fitted else 'N/A', {fitted_str})"
