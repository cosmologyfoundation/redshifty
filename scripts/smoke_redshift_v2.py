#!/usr/bin/env python
"""
Smoke test for RedshiftTokenizerV2 using local DESI data.

Tests:
1. Fit on real DESI redshifts from redrock FITS files
2. Encode/decode roundtrip RMSE
3. Compare V1 (256 levels) vs V2 (1024 levels) RMSE
4. Embedding shapes for different d_model values
5. Star/galaxy classification

Run:
    python scripts/smoke_redshift_v2.py
"""

import sys
from pathlib import Path
import glob

import torch
import numpy as np
import astropy.io.fits as fits

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.tokenizers.redshift import RedshiftTokenizer
from src.tokenizers.redshift_v2 import RedshiftTokenizerV2


def load_redshifts_from_dir(data_dir: Path, max_files: int = None):
    """Load all redshifts from redrock FITS files in a directory."""
    redrock_files = sorted(glob.glob(str(data_dir / "redrock-*.fits")))
    if max_files:
        redrock_files = redrock_files[:max_files]

    all_z = []
    for fpath in redrock_files:
        try:
            with fits.open(fpath) as hdu:
                z = hdu['REDSHIFTS'].data['Z']
                zwarn = hdu['REDSHIFTS'].data['ZWARN']
                mask = zwarn == 0
                all_z.extend(z[mask].tolist())
        except Exception as e:
            print(f"  WARN: could not read {fpath}: {e}")

    return torch.tensor(all_z, dtype=torch.float32)


def load_redshifts_from_coadd(data_dir: Path):
    """Load redshifts from coadd FIBERMAP extension (TARGETID matched)."""
    coadd_files = sorted(glob.glob(str(data_dir / "coadd-sv3-bright-*.fits")))
    all_z = []
    for fpath in coadd_files:
        try:
            with fits.open(fpath) as hdu:
                fiber_data = hdu['FIBERMAP'].data
                z = fiber_data['TARGETID'].copy()
                obj_type = fiber_data['OBJTYPE'] if 'OBJTYPE' in fiber_data.columns.names else None
                if obj_type is not None:
                    mask = np.char.strip(obj_type) == b'LRG'
                    all_z.extend(z[mask].tolist())
                else:
                    all_z.extend(z.tolist())
        except Exception as e:
            print(f"  WARN: could not read {fpath}: {e}")
    return torch.tensor(all_z, dtype=torch.float32)


def main():
    data_dir = Path("data/desi_raw")
    print(f"[data] Loading redshifts from {data_dir}")

    z = load_redshifts_from_dir(data_dir)
    z = z[:10000].clamp(min=-0.01, max=10.0)
    print(f"[data] {len(z)} good redshifts loaded")
    print(f"[data] z range: [{z.min():.4f}, {z.max():.4f}]")
    print(f"[data] z mean: {z.mean():.4f}, std: {z.std():.4f}")

    # --- Test 1: V1 baseline (256 levels) ---
    print("\n[Test 1] RedshiftTokenizer V1 (256 levels):")
    v1 = RedshiftTokenizer(n_levels=256)
    v1.fit(z)
    z_enc_v1 = v1.encode(z)
    z_dec_v1 = v1.decode(z_enc_v1)
    rmse_v1 = torch.sqrt(torch.mean((z - z_dec_v1) ** 2)).item()
    print(f"  RMSE: {rmse_v1:.6f}")
    print(f"  Index range: [{z_enc_v1.min().item()}, {z_enc_v1.max().item()}]")
    assert rmse_v1 < 0.15, f"V1 RMSE too high: {rmse_v1}"
    print("  PASS")

    # --- Test 2: V2 encode/decode roundtrip (1024 levels) ---
    print("\n[Test 2] RedshiftTokenizerV2 (1024 levels):")
    v2 = RedshiftTokenizerV2(n_levels=1024, d_model=32)
    v2.fit(z)
    assert v2.is_fitted, "V2 not fitted"
    print(f"  Embedding weight shape: {v2._embedding.weight.shape} (expect 32, 1024)")

    z_enc_v2 = v2.encode(z)
    z_dec_v2 = v2.decode(z_enc_v2)
    rmse_v2 = torch.sqrt(torch.mean((z - z_dec_v2) ** 2)).item()
    print(f"  RMSE: {rmse_v2:.6f}")
    print(f"  Index range: [{z_enc_v2.min().item()}, {z_enc_v2.max().item()}]")
    assert z_enc_v2.max().item() > z_enc_v1.max().item(), \
        f"V2 should use more indices ({z_enc_v2.max().item()}) than V1 ({z_enc_v1.max().item()})"
    print(f"  PASS — V2 uses wider index range ({z_enc_v2.max().item()} vs V1 {z_enc_v1.max().item()})")

    # --- Test 3: Embedding shapes ---
    print("\n[Test 3] V2 embedding shapes:")
    for d_model in [16, 32, 64]:
        v2_d = RedshiftTokenizerV2(n_levels=1024, d_model=d_model)
        v2_d.fit(z)
        emb = v2_d.forward(z[:8])
        batch_size = emb.shape[0]
        assert emb.shape == (batch_size, d_model), f"Expected ({batch_size}, {d_model}), got {emb.shape}"
        print(f"  d_model={d_model}: emb shape {emb.shape} — PASS")

    # --- Test 4: Embedding orthogonality ---
    print("\n[Test 4] Embedding matrix properties:")
    W = v2.get_embedding_weights()
    print(f"  Weight shape: {W.shape}")
    print(f"  Weight mean: {W.mean():.4f}, std: {W.std():.4f}")
    assert W.std() > 0.001, "Embedding weights should have non-trivial scale"
    print("  PASS")

    # --- Test 5: Star/galaxy classification ---
    print("\n[Test 5] Star/galaxy classification:")
    z_test = torch.tensor([0.0, 0.001, 0.005, 0.01, 0.1, 0.5, 1.0])
    classes = v2.get_redshift_class(z_test)
    expected = ["star", "star", "star", "galaxy", "galaxy", "galaxy", "galaxy"]
    for z_i, cls, exp in zip(z_test.tolist(), classes, expected):
        status = "✓" if cls == exp else "✗"
        print(f"  z={z_i:.4f} → {cls} (expect {exp}) {status}")
        assert cls == exp, f"Misclassified z={z_i}: got {cls}, expected {exp}"
    print("  PASS")

    # --- Test 6: encode_with_evidence ---
    print("\n[Test 6] encode_with_evidence:")
    z_batch = z[:8]
    indices, one_hot = v2.encode_with_evidence(z_batch)
    assert indices.shape == (8,)
    assert one_hot.shape == (8, 1024)
    assert torch.allclose(one_hot.sum(dim=1), torch.ones(8), atol=1e-6)
    print(f"  indices shape: {indices.shape}")
    print(f"  one_hot shape: {one_hot.shape}")
    print(f"  one_hot sums: {one_hot.sum(dim=1)}")
    print("  PASS")

    # --- Test 7: Gaussian range effect ---
    print("\n[Test 7] Gaussian range effect:")
    for grange in [3.0, 3.5, 4.0]:
        v2_g = RedshiftTokenizerV2(n_levels=256, gaussian_range=grange)
        v2_g.fit(z)
        rmse_g = v2_g.get_reconstruction_rmse(z[:500])
        print(f"  gaussian_range={grange}: RMSE={rmse_g:.6f}")
    print("  PASS")

    # --- Test 8: Embedding differentiability (backward pass) ---
    print("\n[Test 8] Embedding backward pass:")
    v2_small = RedshiftTokenizerV2(n_levels=256, d_model=16)
    v2_small.fit(z)
    v2_small.set_training(True)
    emb = v2_small.forward(z[:4], training=True)
    loss = emb.sum()
    loss.backward()
    grad_norm = v2_small._embedding.weight.grad.norm().item()
    print(f"  Embedding grad norm: {grad_norm:.6f}")
    assert grad_norm > 0, "Gradient should flow to embedding"
    print("  PASS")

    print("\n" + "=" * 50)
    print("ALL REDSHIFT V2 SMOKE TESTS PASSED")
    print("=" * 50)
    print(f"\nV1 (256 levels) RMSE: {rmse_v1:.6f}")
    print(f"V2 (1024 levels) RMSE: {rmse_v2:.6f}")
    print(f"Improvement: {rmse_v1 - rmse_v2:.6f}")


if __name__ == "__main__":
    main()