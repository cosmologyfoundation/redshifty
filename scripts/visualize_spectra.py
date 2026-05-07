"""
Visualize DESI Spectra
======================
Quick script to plot a few spectra from the downloaded DESI data.
Run this to verify the data pipeline works end-to-end.

Usage:
    python scripts/visualize_spectra.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

from src.utils.data import DESISpectrumDataset
from src.utils.plotting import plot_spectrum_grid, plot_redshift_distribution
import numpy as np


def main():
    print("Loading DESI spectra...")
    
    coadd_path = Path("data/desi_raw/coadd-sv3-bright-10016.fits")
    redrock_path = Path("data/desi_raw/redrock-sv3-bright-10016.fits")
    
    if not coadd_path.exists():
        print(f"Error: Coadd file not found at {coadd_path}")
        print("Run data/download_desi.py first to download real DESI data.")
        return 1
    
    # Load dataset
    dataset = DESISpectrumDataset(
        coadd_path=coadd_path,
        redrock_path=redrock_path,
    )
    
    # Get redshift range
    z_min, z_max = dataset.get_redshift_range()
    print(f"\nDataset contains {len(dataset)} spectra")
    print(f"Redshift range: [{z_min:.4f}, {z_max:.4f}]")
    
    # Plot first 9 spectra
    print("\nPlotting sample spectra...")
    n_plot = min(9, len(dataset))
    spectra_to_plot = []
    for i in range(n_plot):
        item = dataset[i]
        spectra_to_plot.append({
            "wavelength": item["wavelength"].numpy(),
            "flux": item["flux"].numpy(),
            "ivar": item["ivar"].numpy(),
            "mask": item["mask"].numpy(),
            "z": item["z"].item(),
        })
    
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    
    plot_spectrum_grid(
        spectra_to_plot,
        ncols=3,
        save_path=output_dir / "sample_spectra.png",
    )
    
    # Plot redshift distribution
    print("Plotting redshift distribution...")
    all_z = [dataset[i]["z"].item() for i in range(len(dataset))]
    plot_redshift_distribution(
        np.array(all_z),
        save_path=output_dir / "redshift_distribution.png",
    )
    
    print(f"\nPlots saved to {output_dir}/")
    print("  - sample_spectra.png")
    print("  - redshift_distribution.png")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
