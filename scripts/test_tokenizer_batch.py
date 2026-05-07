"""Quick test of tokenizer on 186 real DESI spectra."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import torch
import numpy as np
from tqdm import tqdm

from src.tokenizers.spectrum import SpectrumTokenizer
from src.utils.data import DESISpectrumDataset

print("=" * 60)
print("Tokenizer Test: 186 Real DESI Spectra")
print("=" * 60)

# Load all 4 files
coadd_files = sorted(Path('data/desi_raw').glob('coadd-*.fits'))
print(f"\nFiles found: {len(coadd_files)}")

all_spectra = []
for f in coadd_files:
    ds = DESISpectrumDataset(
        coadd_path=f, 
        redrock_path=None, 
        require_good_zwarn=False, 
        require_nonzero_flux=True
    )
    for i in range(len(ds)):
        all_spectra.append(ds[i])

print(f"\nTotal spectra loaded: {len(all_spectra)}")

# Device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {device}")

# Init model
model = SpectrumTokenizer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train for just 10 epochs (quick test)
n_epochs = 10
print(f"\nTraining for {n_epochs} epochs on {len(all_spectra)} spectra...")

for epoch in range(n_epochs):
    epoch_loss = 0
    for spec in tqdm(all_spectra, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False):
        flux = spec['flux'].unsqueeze(0).to(device)
        ivar = spec['ivar'].unsqueeze(0).to(device)
        istd = torch.sqrt(ivar.clamp(min=1e-10))
        x = torch.stack([flux, istd], dim=1)
        
        optimizer.zero_grad()
        recon, loss, _ = model(x)
        loss['total'].backward()
        optimizer.step()
        
        epoch_loss += loss['total'].item()
    
    avg_loss = epoch_loss / len(all_spectra)
    print(f"  Epoch {epoch+1}: avg_loss={avg_loss:.4f}")

# Check token utilization
print("\nChecking token utilization...")
model.eval()
all_indices = []

with torch.no_grad():
    for spec in tqdm(all_spectra, desc="Encoding"):
        flux = spec['flux'].unsqueeze(0).to(device)
        ivar = spec['ivar'].unsqueeze(0).to(device)
        istd = torch.sqrt(ivar.clamp(min=1e-10))
        x = torch.stack([flux, istd], dim=1)
        indices, _ = model.encode(x)
        all_indices.append(indices.cpu().numpy())

all_indices = np.concatenate(all_indices, axis=0)
unique_codes = len(np.unique(all_indices))

print(f"\n{'='*60}")
print(f"Token Statistics:")
print(f"  Total tokens:     {all_indices.size:,}")
print(f"  Unique codes:     {unique_codes} / 1024")
print(f"  Utilization:      {unique_codes/1024*100:.1f}%")
print(f"{'='*60}")

if unique_codes > 50:
    print("✅ SUCCESS: Model is using diverse codes!")
else:
    print("⚠️  WARNING: Codebook usage is low")
