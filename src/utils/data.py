"""
DESI Spectrum Dataset
=====================
PyTorch Dataset for DESI coadded spectra.

Reads DESI coadd FITS files and redrock redshift catalogs,
stitches B/R/Z camera bands into a single spectrum,
and returns PyTorch tensors.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from astropy.io import fits
from typing import Optional, Union, List, Dict


# Standard DESI wavelength grid (for reference)
DESI_WAVE_MIN = 3600.0
DESI_WAVE_MAX = 9824.0  # Slightly extended for coadds


def stitch_bands(
    wavelengths: List[np.ndarray],
    fluxes: List[np.ndarray],
    ivars: List[np.ndarray],
    masks: List[np.ndarray],
) -> Dict[str, np.ndarray]:
    """Stitch B, R, Z bands into a single spectrum.
    
    Handles overlaps by inverse-variance weighted averaging.
    
    Args:
        wavelengths: List of wavelength arrays [B, R, Z]
        fluxes: List of flux arrays [B, R, Z]
        ivars: List of inverse variance arrays [B, R, Z]
        masks: List of mask arrays [B, R, Z]
        
    Returns:
        dict with keys: wavelength, flux, ivar, mask
    """
    # Concatenate all wavelengths and sort
    all_wave = np.concatenate(wavelengths)
    all_flux = np.concatenate(fluxes)
    all_ivar = np.concatenate(ivars)
    all_mask = np.concatenate(masks)
    
    # Sort by wavelength
    sort_idx = np.argsort(all_wave)
    all_wave = all_wave[sort_idx]
    all_flux = all_flux[sort_idx]
    all_ivar = all_ivar[sort_idx]
    all_mask = all_mask[sort_idx]
    
    # Find unique wavelengths (within small tolerance)
    # For overlaps, do inverse-variance weighted average
    unique_waves = []
    weighted_flux = []
    total_ivar = []
    combined_mask = []
    
    i = 0
    while i < len(all_wave):
        # Find all pixels within 0.1 Å of current wavelength
        wave = all_wave[i]
        tol = 0.1
        j = i
        while j < len(all_wave) and abs(all_wave[j] - wave) < tol:
            j += 1
        
        # Weighted average in overlap region
        flux_chunk = all_flux[i:j]
        ivar_chunk = all_ivar[i:j]
        mask_chunk = all_mask[i:j]
        
        # Only use unmasked pixels
        good = ~mask_chunk
        if good.any():
            ivar_sum = ivar_chunk[good].sum()
            if ivar_sum > 0:
                avg_flux = (flux_chunk[good] * ivar_chunk[good]).sum() / ivar_sum
            else:
                avg_flux = flux_chunk[good].mean()
            avg_mask = False
        else:
            avg_flux = flux_chunk.mean()
            ivar_sum = 0.0
            avg_mask = True
        
        unique_waves.append(wave)
        weighted_flux.append(avg_flux)
        total_ivar.append(ivar_sum)
        combined_mask.append(avg_mask)
        
        i = j
    
    return {
        "wavelength": np.array(unique_waves, dtype=np.float32),
        "flux": np.array(weighted_flux, dtype=np.float32),
        "ivar": np.array(total_ivar, dtype=np.float32),
        "mask": np.array(combined_mask, dtype=bool),
    }


class DESISpectrumDataset(Dataset):
    """PyTorch Dataset for DESI coadded spectra.
    
    Args:
        coadd_path: Path to DESI coadd FITS file
        redrock_path: Path to redrock redshift FITS file (optional)
        max_spectra: Maximum number of spectra to load (None = all)
        transform: Optional transform to apply to spectra
    """
    
    def __init__(
        self,
        coadd_path: Union[str, Path],
        redrock_path: Optional[Union[str, Path]] = None,
        max_spectra: Optional[int] = None,
        transform=None,
        require_good_zwarn: bool = True,
        require_nonzero_flux: bool = True,
    ):
        self.coadd_path = Path(coadd_path)
        self.redrock_path = Path(redrock_path) if redrock_path else None
        self.transform = transform
        self.require_good_zwarn = require_good_zwarn
        self.require_nonzero_flux = require_nonzero_flux
        
        # Load spectra from coadd file
        self.spectra = self._load_coadd(max_spectra)
        self.n_spectra = len(self.spectra)
        
        print(f"Loaded {self.n_spectra} good spectra from {self.coadd_path.name}")
        if self.n_spectra > 0:
            print(f"  Wavelength range: [{self.spectra[0]['wavelength'].min():.1f}, "
                  f"{self.spectra[0]['wavelength'].max():.1f}] Å")
            print(f"  Pixels per spectrum: {len(self.spectra[0]['wavelength'])}")
    
    def _load_coadd(self, max_spectra: Optional[int] = None) -> List[Dict]:
        """Load and stitch spectra from coadd FITS file.
        
        Filters out bad spectra based on ZWARN flags, fiber status, and flux.
        """
        spectra = []
        n_filtered_zwarn = 0
        n_filtered_flux = 0
        n_filtered_fiber = 0
        
        with fits.open(self.coadd_path) as hdul:
            n = hdul["B_FLUX"].data.shape[0]
            
            # Read quality flags from coadd fibermap
            fibermap = hdul["FIBERMAP"].data
            fiberstatus = fibermap["COADD_FIBERSTATUS"]
            
            # Read redshifts and quality flags if available
            redshifts = None
            zwarn = None
            if self.redrock_path and self.redrock_path.exists():
                with fits.open(self.redrock_path) as rh:
                    redshifts = rh["REDSHIFTS"].data["Z"]
                    zwarn = rh["REDSHIFTS"].data["ZWARN"]
            
            for i in range(n):
                # Apply filters
                
                # Filter 1: Check fiber status (0 = good)
                if self.require_good_zwarn and fiberstatus[i] != 0:
                    n_filtered_fiber += 1
                    continue
                
                # Filter 2: Check ZWARN (0 = good redshift)
                if self.require_good_zwarn and zwarn is not None and zwarn[i] != 0:
                    n_filtered_zwarn += 1
                    continue
                
                # Read each band
                b_flux = hdul["B_FLUX"].data[i]
                b_ivar = hdul["B_IVAR"].data[i]
                b_mask = hdul["B_MASK"].data[i] != 0  # Convert to bool
                b_wave = hdul["B_WAVELENGTH"].data
                
                r_flux = hdul["R_FLUX"].data[i]
                r_ivar = hdul["R_IVAR"].data[i]
                r_mask = hdul["R_MASK"].data[i] != 0
                r_wave = hdul["R_WAVELENGTH"].data
                
                z_flux = hdul["Z_FLUX"].data[i]
                z_ivar = hdul["Z_IVAR"].data[i]
                z_mask = hdul["Z_MASK"].data[i] != 0
                z_wave = hdul["Z_WAVELENGTH"].data
                
                # Filter 3: Check for non-zero flux
                if self.require_nonzero_flux:
                    total_flux = (
                        np.abs(b_flux).sum() + np.abs(r_flux).sum() + np.abs(z_flux).sum()
                    )
                    if total_flux == 0:
                        n_filtered_flux += 1
                        continue
                
                # Stitch bands
                stitched = stitch_bands(
                    [b_wave, r_wave, z_wave],
                    [b_flux, r_flux, z_flux],
                    [b_ivar, r_ivar, z_ivar],
                    [b_mask, r_mask, z_mask],
                )
                
                # Add redshift
                if redshifts is not None:
                    stitched["z"] = float(redshifts[i])
                else:
                    stitched["z"] = 0.0
                
                spectra.append(stitched)
                
                # Stop if we've reached max_spectra
                if max_spectra is not None and len(spectra) >= max_spectra:
                    break
        
        # Report filtering stats
        total_filtered = n_filtered_zwarn + n_filtered_flux + n_filtered_fiber
        if total_filtered > 0:
            print(f"  Filtered out {total_filtered} bad spectra:")
            if n_filtered_fiber > 0:
                print(f"    {n_filtered_fiber} bad fiber status")
            if n_filtered_zwarn > 0:
                print(f"    {n_filtered_zwarn} bad ZWARN")
            if n_filtered_flux > 0:
                print(f"    {n_filtered_flux} zero flux")
        
        return spectra
    
    def __len__(self):
        return self.n_spectra
    
    def __getitem__(self, idx):
        spec = self.spectra[idx]
        
        item = {
            "flux": torch.from_numpy(spec["flux"].copy()),
            "ivar": torch.from_numpy(spec["ivar"].copy()),
            "mask": torch.from_numpy(spec["mask"].copy()),
            "wavelength": torch.from_numpy(spec["wavelength"].copy()),
            "z": torch.tensor(spec["z"], dtype=torch.float32),
        }
        
        if self.transform:
            item = self.transform(item)
        
        return item
    
    def get_redshift_range(self):
        """Get the redshift range of the dataset."""
        zs = [s["z"] for s in self.spectra]
        return min(zs), max(zs)


def collate_desi_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader that pads spectra to same length.
    
    Since all DESI spectra should have the same wavelength grid,
    this is mostly a simple stack, but handles any edge cases.
    """
    return {
        "flux": torch.stack([b["flux"] for b in batch]),
        "ivar": torch.stack([b["ivar"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "wavelength": torch.stack([b["wavelength"] for b in batch]),
        "z": torch.stack([b["z"] for b in batch]),
    }
