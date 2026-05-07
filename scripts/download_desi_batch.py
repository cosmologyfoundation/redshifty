"""
Download multiple DESI healpix files for expanded testing.

Usage:
    python scripts/download_desi_batch.py --n-files 10
"""

import requests
import re
from pathlib import Path
from tqdm import tqdm
import argparse


BASE_URL = "https://data.desi.lbl.gov/public/edr/spectro/redux/fuji/healpix/sv3/bright"


def list_available_pixels():
    """List all available healpix pixels."""
    pixels = []
    
    # Get top-level directories
    r = requests.get(BASE_URL + "/", timeout=30)
    if r.status_code != 200:
        return pixels
    
    dirs = re.findall(r'href="(\d+)/"', r.text)
    
    for d in dirs:
        # Get subdirectories
        r2 = requests.get(f"{BASE_URL}/{d}/", timeout=30)
        if r2.status_code != 200:
            continue
        
        subdirs = re.findall(r'href="(\d+)/"', r2.text)
        for sub in subdirs:
            pixels.append((d, sub))
    
    return pixels


def download_pixel(hp_dir, hp_pixel, output_dir, overwrite=False):
    """Download a single healpix pixel."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"coadd-sv3-bright-{hp_pixel}.fits"
    url = f"{BASE_URL}/{hp_dir}/{hp_pixel}/{filename}"
    filepath = output_dir / filename
    
    if filepath.exists() and not overwrite:
        return filepath
    
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        with open(filepath, "wb") as f:
            f.write(r.content)
        return filepath
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-files", type=int, default=10, help="Number of healpix files to download")
    parser.add_argument("--output-dir", default="data/desi_raw", help="Output directory")
    args = parser.parse_args()
    
    print("Listing available DESI healpix pixels...")
    pixels = list_available_pixels()
    print(f"Found {len(pixels)} total pixels")
    
    if len(pixels) == 0:
        print("No pixels found!")
        return
    
    # Download first n files
    n = min(args.n_files, len(pixels))
    print(f"\nDownloading {n} healpix files...")
    
    downloaded = []
    for hp_dir, hp_pixel in tqdm(pixels[:n], desc="Downloading"):
        filepath = download_pixel(hp_dir, hp_pixel, args.output_dir)
        if filepath:
            downloaded.append(filepath)
    
    print(f"\nDownloaded {len(downloaded)} files to {args.output_dir}/")
    total_size = sum(f.stat().st_size for f in downloaded) / (1024 * 1024)
    print(f"Total size: {total_size:.1f} MB")


if __name__ == "__main__":
    main()
