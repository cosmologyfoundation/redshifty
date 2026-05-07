"""
Quick test of tokenizer with normalization.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.tokenizers.spectrum import SpectrumTokenizer
from src.utils.data import DESISpectrumDataset

# Load data
coadd_path = Path("data/desi_raw/coadd-sv3-bright-10016.fits")
redrock_path = Path("data/desi_raw/redrock-sv3-bright-10016.fits")
dataset = DESISpectrumDataset(coadd_path=coadd_path, redrock_path=redrock_path)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Init model
tokenizer = SpectrumTokenizer().to(device)
optimizer = torch.optim.Adam(tokenizer.parameters(), lr=1e-3)

# Train
n_epochs = 20
losses = []

print("Training with normalization...")
for epoch in range(n_epochs):
    epoch_loss = 0
    for i in range(len(dataset)):
        spec = dataset[i]
        flux = spec["flux"].unsqueeze(0).to(device)
        ivar = spec["ivar"].unsqueeze(0).to(device)
        istd = torch.sqrt(ivar.clamp(min=1e-10))
        x = torch.stack([flux, istd], dim=1)  # (1, 2, L)
        
        optimizer.zero_grad()
        recon, loss, _ = tokenizer(x)
        loss["total"].backward()
        optimizer.step()
        
        epoch_loss += loss["total"].item()
    
    avg_loss = epoch_loss / len(dataset)
    losses.append(avg_loss)
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d}: loss={avg_loss:.4f}")

print(f"Final loss: {losses[-1]:.4f}")

# Evaluate
tokenizer.eval()
all_mse = []
all_r2 = []

with torch.no_grad():
    for i in range(len(dataset)):
        spec = dataset[i]
        flux = spec["flux"].unsqueeze(0).to(device)
        ivar = spec["ivar"].unsqueeze(0).to(device)
        istd = torch.sqrt(ivar.clamp(min=1e-10))
        x = torch.stack([flux, istd], dim=1)
        
        recon, _, _ = tokenizer(x)
        
        # Interpolate recon to original length for fair comparison
        flux_orig = flux[0].cpu().numpy()
        flux_recon = recon[0, 0].cpu().numpy()
        
        # Interpolate
        wave_orig = np.arange(len(flux_orig))
        wave_recon = np.linspace(0, len(flux_orig)-1, len(flux_recon))
        flux_recon_interp = np.interp(wave_orig, wave_recon, flux_recon)
        
        mse = np.mean((flux_orig - flux_recon_interp) ** 2)
        ss_res = np.sum((flux_orig - flux_recon_interp) ** 2)
        ss_tot = np.sum((flux_orig - np.mean(flux_orig)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        all_mse.append(mse)
        all_r2.append(r2)

print(f"\nReconstruction Quality (WITH normalization):")
print(f"  Mean MSE: {np.mean(all_mse):.4f} ± {np.std(all_mse):.4f}")
print(f"  Mean R²:  {np.mean(all_r2):.3f} ± {np.std(all_r2):.3f}")
print(f"  Best R²:  {np.max(all_r2):.3f}")
print(f"  Worst R²: {np.min(all_r2):.3f}")

# Plot first spectrum
spec = dataset[0]
flux = spec["flux"].unsqueeze(0).to(device)
ivar = spec["ivar"].unsqueeze(0).to(device)
istd = torch.sqrt(ivar.clamp(min=1e-10))
x = torch.stack([flux, istd], dim=1)

with torch.no_grad():
    recon, _, _ = tokenizer(x)

flux_orig = flux[0].cpu().numpy()
flux_recon = recon[0, 0].cpu().numpy()
wave_orig = spec["wavelength"].cpu().numpy()
wave_recon = np.linspace(wave_orig.min(), wave_orig.max(), len(flux_recon))

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(wave_orig, flux_orig, color='black', linewidth=0.8, label='Original')
ax.plot(wave_recon, flux_recon, color='red', linewidth=0.8, alpha=0.7, label='Reconstructed')
ax.set_xlabel('Wavelength [Å]')
ax.set_ylabel('Flux')
ax.set_title(f'z = {spec["z"].item():.4f} | With Normalization | R² = {all_r2[0]:.3f}')
ax.legend()
plt.tight_layout()
plt.savefig('plots/tokenizer_with_norm.png', dpi=150)
print("\nSaved plot to plots/tokenizer_with_norm.png")
