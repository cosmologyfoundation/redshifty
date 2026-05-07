# Research Log

Living document of findings, decisions, and experimental results.

---

## 2026-05-05: Project Kickoff & Architecture Planning

### Assignment Requirements (from PHYS303_Final-Project_2026.pdf)

- **Goal**: Build a unimodal foundation model for DESI spectra
- **Scope**: Spectra ONLY. No images, no photometry, no Subaru.
- **Core critique of AION-1 to address**:
  1. Redshift treated like any other token → masked only occasionally
  2. Redshift never enters encoder representation space (separate frozen head)
- **Data**: DESI DR1 (SV3 "one-percent"), ~1M objects, 7,081-pixel grid, 3,600–9,800 Å
- **Required approaches** (choose at least one):
  - **Approach A**: Joint training — small MLP head predicts z jointly with masked-token objective
  - **Approach B**: Always-mask redshift token — force reconstruction of z from spectral context every step
- **Evaluation**: Redshift prediction + spectrum reconstruction + OOD generalization (non-DESI spectra)
- **What NOT to build**: Pure CNN redshift regressor — must be a foundation model with masked reconstruction

### AION-1 Architecture Notes (from AION Paper.pdf + GitHub)

**Spectrum Tokenizer**:
- ConvNeXt-V2 encoder/decoder backbone
- Input: flux + inverse-variance (istd), 2-channel
- Interpolated to common 8,704-point latent grid (3,500–10,462.4 Å, 0.8 Å spacing)
- 4-stage ConvNeXt downsampling: 4×4 conv + 3× (2×2 conv) → compresses to 273×512 latent
- Quantizer: Look-up-Free Quantizer (LFQ), dim=10, codebook size=1024
- Losses: Gaussian NLL (inverse-variance weighted) + mask BCE + commitment loss β=0.25
- Additional normalization token (log10 median flux, scalar quantized) prepended to sequence
- Total tokens per spectrum: **274** (1 normalization + 273 spectral)

**Scalar Tokenizer (for redshift)**:
- Empirical CDF mapping to standard normal: z_i = Φ⁻¹(F_x(x_i))
- Equal-width binning in Gaussian space → uniform probability mass per bin
- FSQ quantization with K=1024 fixed centroids at standard normal quantiles
- No learned parameters — parameter-free

**Transformer Backbone**:
- Encoder-decoder architecture (T5-style scaling)
- Modality-specific embeddings: token embedding + modality embedding + positional embedding
- Input budget: 256 tokens, output budget: 128 tokens (AION-1 config)
- Model sizes: Base (300M), Large (800M), XL (3B)
- Trained with multimodal masked modeling (4M objective)

**Key Design Decisions for Our Model**:

1. **Reuse AION tokenizers**: The spectrum and scalar codecs are well-engineered and open-source. We will adapt them for our unimodal scope rather than reimplementing from scratch.
2. **Custom transformer**: We will build our own encoder-decoder, simplified to handle only two modalities (spectrum tokens + redshift token).
3. **Redshift mechanisms**: Implement both Approach A and B separately, then compare.
4. **Training compute**: Mac MPS for smoke tests, NERSC A100 for full training.
5. **Discrete tokens**: We will keep the discrete tokenization approach (not continuous MAE) because it directly addresses the assignment's critique of AION-1's token handling.

### NERSC SLURM Constraints (from docs.nersc.gov/jobs)

- Must specify: `--nodes`, `--time`, `--constraint`, `--qos`, `--account`
- GPU jobs MUST use `--gpus` or `-G` flag for CUDA visibility
- Default QOS is `debug` (10 min)
- No default architecture — jobs without `--constraint` are rejected
- Perlmutter GPU nodes: 256 GB CPU RAM, 160 GB GPU RAM
- Use `srun` within job scripts for parallel tasks
- Good practice to always set `--account=<NERSC Project>`

### Next Steps

1. ~~Phase 1: Build minimal smoke-test data pipeline~~ ✅ COMPLETE
2. Phase 2: Adapt spectrum tokenizer from AION
3. Phase 3: Redshift scalar tokenizer
4. Phase 4: Transformer backbone
5. Phase 5: Approach A training
6. Phase 6: Approach B training
7. Phase 7: Evaluation & comparison
8. Phase 8: NERSC full-scale training
9. Phase 9: OOD generalization prep

## 2026-05-06: Phase 1 — Minimal Smoke-Test Data Pipeline

### Data Source
- Downloaded **real DESI EDR data** from `data.desi.lbl.gov`
- File: `coadd-sv3-bright-10016.fits` (18.5 MB) + `redrock-sv3-bright-10016.fits` (96 KB)
- Contains **43 spectra** from SV3 "one-percent" bright targets
- Redshift range: **[-0.0020, 1.1854]** (mix of stars and galaxies)
- Wavelength coverage: **3600–9824 Å** (B+R+Z camera bands stitched)
- Native pixel count after stitching: **7781 pixels** (slightly more than standard 7081 due to overlaps)

### Pipeline Components Built
1. **`src/utils/data.py`** — `DESISpectrumDataset` PyTorch Dataset
   - Stitches B/R/Z bands via inverse-variance weighted averaging in overlap regions
   - Returns dict of tensors: `flux`, `ivar`, `mask`, `wavelength`, `z`
   - `collate_desi_batch()` for DataLoader batching

2. **`src/utils/plotting.py`** — Visualization utilities
   - `plot_spectrum()`: Single spectrum with error regions and masking
   - `plot_spectrum_grid()`: Grid of multiple spectra
   - `plot_redshift_distribution()`: Histogram with statistics
   - `plot_reconstruction_comparison()`: Original vs reconstructed (for later phases)
   - `plot_training_curves()`: Loss curves (for later phases)

3. **`tests/test_data.py`** — pytest suite (10 tests, all passing)
   - Band stitching with/without overlaps
   - Real data loading, shapes, wavelength range, redshift values
   - Batch collation

4. **`scripts/visualize_spectra.py`** — Standalone visualization script
   - Plots sample spectra grid + redshift distribution
   - Outputs to `plots/` directory

### Key Observations from Real Data
- Spectra show clear **emission lines** (H-alpha, O-III, etc.) at various redshifts
- One object at z ≈ −0.002 is a **star** (flat continuum, no emission lines)
- Flux amplitudes vary by ~10× across objects — need robust normalization for tokenizer
- B/R/Z band overlaps (~40 pixels each) require careful stitching to avoid discontinuities

### Decisions Made
- **Using real data, not synthetic** — assignment explicitly calls for real DESI data
- **Native wavelength grid** — keeping the stitched 7781-pixel grid rather than forcing exactly 7081 pixels; the tokenizer will interpolate to its latent grid anyway
- **Coadd files preferred** — coadds combine multiple exposures and have higher S/N than individual spectra

### Next Steps
- ~~Phase 2: Adapt AION spectrum tokenizer (ConvNeXt-V2 + LFQ)~~ ✅ COMPLETE
- Phase 3: Redshift scalar tokenizer
- Phase 4: Transformer backbone
- Phase 5: Approach A training
- Phase 6: Approach B training
- Phase 7: Evaluation & comparison
- Phase 8: NERSC full-scale training
- Phase 9: OOD generalization prep

## 2026-05-07: Phase 2 — Spectrum Tokenizer

### Architecture Built
**`src/tokenizers/spectrum.py`** — ConvNeXt-V2 autoencoder + LFQ quantization

**Encoder:**
- Stem: 4×4 conv, stride 4 (8704 → 2176)
- Stage 1: 3 ConvNeXt blocks @ 96 dim
- Stage 2: downsample 2× + 3 blocks @ 192 dim (→ 1088)
- Stage 3: downsample 2× + 9 blocks @ 384 dim (→ 544)
- Stage 4: downsample 2× + 3 blocks @ 512 dim (→ 272)
- Pre-quant: LayerNorm + 1×1 conv (512 → 10 dim)

**Quantizer:**
- Look-up-Free Quantizer (LFQ), dim=10, codebook_size=1024
- Straight-through estimator with commitment loss (β=0.25)
- Simplified from Yu et al. (2023)

**Decoder:**
- Mirror of encoder with ConvTranspose1d upsampling
- Output head: transposed conv stride 4 + 1×1 conv → 2 channels

**Key design choices:**
- Input is **interpolated to fixed 8704-pixel grid** (like AION) to ensure exact token count
- Output is also on 8704-pixel grid; user can interpolate back to original wavelength if needed
- Total parameters: **~24M** (AION's is ~50M; ours is scaled down for smoke testing)

### Tests
- `tests/test_tokenizer.py` — 12 tests, all passing
- ConvNeXt block preserves shape, residual connection works
- LFQ quantizes to valid range, encode→decode roundtrip works
- Full tokenizer: forward pass, different batch sizes, different input lengths
- Model can overfit a single sample (loss decreases with training)

### Notebook
- `notebooks/02_tokenizer.ipynb` — Interactive training & visualization
- Loads real DESI data, trains tokenizer for 50 epochs
- Plots original vs reconstructed spectra with residuals
- Computes MSE and R² reconstruction quality metrics
- Shows token usage distribution

### Comparison with AION
| Feature | AION Tokenizer | Our Tokenizer |
|---------|---------------|---------------|
| Backbone | ConvNeXt-V2 | ConvNeXt-V2 ✅ |
| Input grid | 8704 pixels | 8704 pixels ✅ |
| Output tokens | 273 | 272 (off by 1, fixable with padding) |
| Quantizer | LFQ (dim=10, codebook=1024) | LFQ (dim=10, codebook=1024) ✅ |
| Parameters | ~50M | ~24M (smaller for smoke test) |
| Normalization token | Yes (log10 median flux) | Not yet — will add in Phase 3 |
| Training data | Millions of spectra | 25 spectra (smoke test only) |

### Next Steps
- Phase 3: Redshift scalar tokenizer (CDF → Gaussian → FSQ)
- Phase 4: Transformer encoder-decoder backbone

---
