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

## 2026-05-08: Phase 8 — NERSC Scaffolding for Tokenizer Pretrain

### Diagnosis: Why local results stalled

After running both Approach A and B for 10 epochs on 269 spectra with the
"fixed_split" honest random validation (seed=42), val loss bottomed out at
3.44 (A) / 3.09 (B) and overall accuracy at ~20–24%. Redshift accuracy
plateaued at 84.5% — that's the star-prior shortcut, not real learning.

The dominant cause is **the spectrum tokenizer is still random-init**. The
transformer is being trained against essentially random discrete codes, so
spectrum_acc is structurally bounded near the noise floor regardless of
model size or epoch count. This must be fixed before any further
transformer scaling experiments.

### Decision: NERSC first, tokenizer first

Move to Perlmutter for compute. Pretrain the tokenizer end-to-end on real
DR1 (not just the 4 healpix files we have locally), then re-run the
transformer with the trained codebook frozen.

### NERSC environment facts (verified)

- DR1 lives at `/global/cfs/cdirs/desi/public/dr1`, world-readable to any
  NERSC user. No `desi` unix group needed for the public tree.
- DR1 production = **iron**. Healpix coadds at
  `spectro/redux/iron/healpix/{survey}/{program}/{hpix_group}/{healpix}/`.
  Surveys: sv1/sv2/sv3/main. Programs: bright/dark/backup/other.
- Authoritative redshift catalog: `zcatalog/v1/zall-pix-iron.fits` (~21 GB);
  per-program `zpix-{survey}-{program}.fits` is the right file for subset
  selection without globbing.
- NERSC project name: **`deepsrch`**. GPU jobs use the **`_g`** suffix:
  `--account=deepsrch_g`. CPU jobs are bare `deepsrch`.
- QOS choice: **`shared`** lets us request `--gpus=1` and pay 1/4 the
  allocation hours vs `regular` (which forces a full 4-GPU node). Up to
  48h wallclock. Right call for a single-GPU pretrain.
- Filesystem: code in `$HOME`, manifests/checkpoints in `$SCRATCH`
  (high-perf Lustre but **purged after ~8 weeks idle**), final artifacts
  mirrored back to `$CFS` / repo `checkpoints/nersc/`.

### Architecture: manifest-based streaming, not preload

The local `DESISpectrumDataset` loads every spectrum at `__init__` time.
That doesn't scale to DR1's millions of spectra. Solution:

1. `nersc/build_dr1_index.py` walks the iron tree once, writes a JSONL
   manifest of `(coadd_path, redrock_path, n_rows, survey, program,
   healpix)` records.
2. `nersc/dr1_dataset.py::DR1IndexedDataset` flattens the manifest to one
   `(rec_idx, row_idx)` per spectrum. `__getitem__` opens FITS on demand
   with a small memmap'd HDUL cache. Multi-worker DataLoader parallelizes
   I/O.
3. `collate_dr1_skip_none` drops rows that fail ZWARN/fiber-status/flux
   filters at read time, so quality cuts apply naturally.

### Training entry point

`nersc/pretrain_tokenizer.py`:
- Single-GPU AMP loop. AdamW + cosine schedule with warmup.
- Reuses `SpectrumTokenizer.forward` which already returns
  `{total, recon, quant}` losses. We backprop on `total`.
- Periodic checkpointing to `$SCRATCH/deepsrch/checkpoints/<run>/`,
  best/final mirrored to `--cfs-out` for `$SCRATCH`-purge survival.
- Smoke flag: 50 steps, 200 spectra, no AMP — validates the pipeline in
  a few minutes inside the 10-min `shared`-QOS smoke job.

DDP intentionally deferred. `nersc/ddp_template.slurm` is a placeholder;
promoting the trainer to DDP is a small, separate code change (wrap in
DDP, swap shuffle for DistributedSampler) that should happen *after* the
single-GPU run validates the data path.

### Submission flow

```
ssh perlmutter
cd ~/FoundationModel
bash nersc/setup_env.sh                # one-time
sbatch nersc/smoke_tokenizer.slurm     # 10 min, ~few hundred spectra
sbatch nersc/pretrain_tokenizer.slurm  # 24h, ~hundreds of thousands of spectra
```

### Backlog / ideas

- **Top-hat 5-pixel convolution** as preprocessing before the tokenizer
  encoder — may smooth out per-pixel noise that the LFQ codebook is
  currently spending capacity on. Try this once the baseline tokenizer
  is trained, as an ablation.
- Once `best.pt` exists from NERSC: add `--tokenizer-ckpt PATH` to
  `scripts/train.py` so the transformer training (Approach A and B)
  loads frozen pretrained weights instead of random init.
- Honest val for transformer should be a held-out set of *healpix*
  files, not random rows from the same healpix — eliminates
  same-pointing leakage.

### Next steps

- Phase 8 (in flight): tokenizer pretrain on Perlmutter shared GPU.
- Phase 9: re-run Approach A and B with frozen pretrained tokenizer at
  `d_model=768, n_layers=6` on full DR1 (sv3 + main, bright + dark).
- Phase 10: OOD generalization on non-DESI spectra (assignment
  requirement).
