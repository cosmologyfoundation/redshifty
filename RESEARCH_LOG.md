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

---

## 2026-05-08: Phase 8 Result — Tokenizer Pretrain

### Trial run (job 52693687)

- 200 healpix files (sv3+bright), 393967 row index → 200-spectrum cap.
- 20k steps, batch 32, single A100 in `shared` QOS, 5h18min wallclock.
- I/O bound: 1.05 step/s × batch 32 = 34 spec/s — well below A100's
  ~200-500 spec/s for this 24M-param model. CFS FITS-read bandwidth is
  the bottleneck.

### Loss curve

| step | val_recon | val_total | val_quant |
|---|---|---|---|
| 500 | 6.63 | 6.68 | 0.05 |
| 2000 | 2.46 | 2.68 | 0.22 |
| 5000 | 1.73 | 2.04 | 0.31 |
| 12500 | 1.43 | 1.74 | 0.31 |
| **16500** | **1.35** | **1.69** | 0.34 ← best |
| 19500 | 1.34 | 1.79 | 0.45 |

Smooth descent to a clear plateau by ~step 14000. `quant_loss` climbing
from 0.05 → 0.4 is expected — the codebook starts being used (initial
collapse → diversified codes).

**Outcome:** `best.pt` at step 16500 is a real tokenizer. Used as the
frozen tokenizer for all subsequent transformer experiments.

### Performance opportunities (not pursued yet)

- Staging healpix files from CFS → SCRATCH before training: ~5-10×
  step-rate speedup (CFS read is the dominant cost).
- DDP across 4 GPUs in `regular` QOS: ~3-3.5× wallclock speedup, but
  also 4× allocation cost — only worth it after staging fixes I/O.
- 24h budget needs ~1.2 step/s for 100k steps; we got 0.23 step/s at
  100M-param transformer scale (next phase).

---

## 2026-05-11: Phase 9 Trial — Transformer with Frozen Pretrained Tokenizer

### Pre-fix runs (jobs 52827566 / 52827575, no redshift weighting)

Both Approach A and B at full scale (`d_model=768`, 6 layers, 100M params,
batch 8, AMP). 200 healpix, 393967 spectra, val_frac=0.02. 20k step
budget; jobs hit the 6h wallclock at step ~9000.

| step | A spec_acc | B spec_acc | A z_acc | B z_acc |
|---|---|---|---|---|
| 1500 | 38.8% | 39.6% | 2.4% | 0.7% |
| 5000 | 47.2% | 45.5% | 1.3% | 1.3% |
| 6000 | 55.7% | 65.1% | 3.0% | 0.0% |
| 7000 | 88.4% | **98.8%** | 3.2% | 2.7% |
| 9000 | **97.2%** | **99.91%** | 4.1% | 2.0% |

**Major discovery: spec_acc → 99% is the model learning a trivial
cross-attention copy, not real spectrum modeling.**

For Approach B, decoder at position `j` (`j≥1`) predicts spectrum
token `s_j`. The encoder is `[SOS, s1, ..., sN, EOS]` and cross-attention
is unmasked. The model learns to attend from decoder position `j` to
encoder position `j` — an off-by-one shift, pure copy. No spectroscopy
involved. The step 5500→7000 explosion (50% → 99%) is when this attention
pattern is discovered.

For Approach A, encoder is `[SOS, redshift, s1, ..., sN, EOS]`, off-by-two
shift, slightly harder but same trick. Hence A's 97% ceiling vs B's 99.9%.

**Critically, `z_acc` stayed at random (~2-4%) the whole time.** Reason:
cross-entropy averaged over 274 sequence positions, redshift contributes
1/274 ≈ 0.4% of gradient signal. The model has zero incentive to learn
position 0, so it never does.

**Conclusion:** the unweighted runs don't test the project's thesis.
Both Approach A and B degenerate into the same trivial copy-from-encoder
behavior, and neither learns redshift. We need to force position 0 to
matter.

### Fix 1: Redshift loss weighting

`forward()` now accepts `redshift_weight: float = 1.0` and splits the
cross-entropy by position:
```
loss = redshift_weight * loss_redshift_mean + loss_spectrum_mean
```
At `--redshift-loss-weight 50` (current default), redshift's aggregate
gradient share is 50:1 vs spectrum (~98% of total loss mass at position 0).

`compute_loss_breakdown` helper added to `src/training/utils.py` to log
unweighted per-segment losses separately — visible in `metrics.jsonl`
and wandb so we can tell whether the redshift term is actually
descending.

### Fix 2: Weights & Biases integration

`nersc/_wandb_util.py` provides `init_wandb` / `wlog` / `wfinish`
helpers used by both `pretrain_tokenizer.py` and `train_transformer.py`.
Reads `WANDB_API_KEY` from `.env` (gitignored) via `python-dotenv`.
Modes: online / offline / disabled. NERSC compute nodes default to
offline; metrics are written locally and uploaded later from a login
node with `wandb sync`.

### Weighted run (jobs 52840231 / 52840234, redshift_weight=50)

Same data, same model size, weight=50. First impression at step 1500
looked like the weighting was starving spectrum learning (15% spec_acc
vs the unweighted run's 38.8% at the same step). But the full
trajectory tells a different story:

**Approach A:**

| step | spec_acc | z_acc | loss_redshift |
|---|---|---|---|
| 1500 | 15.6% | 1.5% | 4.90 |
| 3000 | 23.6% | 1.5% | 4.81 |
| 4500 | 25.7% | 1.0% | 4.80 |
| 5000 | 26.4% | 3.1% | 4.65 |
| **5500** | **26.7%** | **5.8%** | **4.19** ← cross-attention copy discovered |

**Approach B:**

| step | spec_acc | z_acc | loss_redshift |
|---|---|---|---|
| 1500 | 11.2% | 0.3% | 4.77 |
| 3000 | 20.4% | 1.4% | 4.75 |
| 4000 | 22.8% | 2.95% | 4.70 |

Between step 4500 and 5500, A's z_acc jumped 1% → 5.8% and
`loss_redshift` dropped from 4.80 → 4.19 — the first real descent since
warmup. This is the model finally discovering the cross-attention
redshift pathway (for A it's a copy; for B it has to extract from
spectrum encoding).

### Why initial diagnosis was wrong

At step 1500 the weighted run looked worse on every axis. The temptation
was to lower the weight immediately. But the model needs time to discover
the cross-attention copy for redshift (~5000 steps with weight=50). The
unweighted run hit 50% spec_acc fast because the spectrum copy is easier
to discover than the redshift one — partly because redshift gets so
little gradient.

Lesson: don't kill a run during the "boring middle" phase of training.
The interesting behavior often emerges after a long flat period.

### Open questions / next moves

1. **How high does the weighted run's `z_acc` go?** A's 5.8% at step 5500
   should keep climbing fast (it's a copy task). B's progression is the
   real test — does the encoder actually encode redshift into its
   hidden state from spectrum alone?

2. **Does the weighted run also reach >90% `spec_acc` eventually?**
   Or does the heavy redshift focus permanently slow spectrum learning?

3. **Is weight=50 the right value?** Could try `weight ∈ {5, 10, 20}`
   if z_acc plateaus too low or spec_acc stays starved.

4. **Held-out healpix split.** Current val set is random rows from the
   same healpix files as train. For honest generalization we should
   hold out entire healpix files.

5. **Top-hat 5-pixel convolution as tokenizer preprocessing.** Still on
   the backlog; would smooth pixel noise the LFQ codebook is spending
   capacity on.

### Files touched this phase

- `src/models/transformer.py` — `redshift_weight` kwarg + position-split loss
- `src/training/utils.py` — `compute_loss_breakdown` helper
- `nersc/_wandb_util.py` (new) — wandb init/log/finish helpers
- `nersc/train_transformer.py` — flags, threading, breakdown + wandb logging
- `nersc/pretrain_tokenizer.py` — wandb logging
- `nersc/dr1_dataset.py`, `nersc/dr1_tokenized_dataset.py` — manifest-based DR1 loaders
- `nersc/train_transformer.slurm`, `nersc/smoke_transformer.slurm` — SLURM entry points
- `requirements.txt`, `pyproject.toml`, `nersc/setup_env.sh` — `python-dotenv` added
- `nersc/README.md` — wandb + weighting documentation

---

## 2026-05-12: Phase 9 Final Result — Thesis Tested

### Setup

- Two 6h runs in parallel: jobs 52840231 (Approach A) / 52840234 (Approach B).
- `redshift_loss_weight=50` (per Phase 9 fix).
- 200 healpix files (sv3+main, bright+dark), 393967 spectra in flat index.
- Frozen pretrained tokenizer from Phase 8 (`tokenizer_v1_52693687/best.pt`, val_recon 1.35).
- 100M parameter transformer: `d_model=768`, 6 encoder + 6 decoder layers, 12 heads, AMP on.
- AdamW, lr=2e-4, cosine schedule with 1000-step linear warmup. batch=8.
- Throughput ~0.55 step/s — both jobs hit the 6h wallclock somewhere around step 10000–12000.

### Result: A learns, B stays at random

**Approach A** discovered the cross-attention "copy redshift from encoder" pathway at step ~6500 and z_acc climbed steeply from there:

| step | val_redshift_acc | val_loss_redshift | val_spectrum_acc | val_loss_spectrum |
|---|---|---|---|---|
| 500 | 0.6% | 5.10 | 0.0% | 6.64 |
| 1500 | 1.5% | 4.90 | 15.6% | 3.70 |
| 3000 | 1.5% | 4.81 | 23.6% | 3.07 |
| 4500 | 1.0% | 4.80 | 25.7% | 2.88 |
| 5500 | 5.8% | 4.19 | 26.7% | 2.82 |
| **6500** | **18.4%** | **3.65** | 27.0% | 2.80 |
| 8000 | 29.3% | 2.94 | 27.9% | 2.75 |
| 9000 | 36.7% | 2.61 | 28.2% | 2.73 |
| 10000 | 42.4% | 2.44 | 28.5% | 2.71 |
| 11000 | 49.2% | 2.18 | 28.9% | 2.68 |
| **11500** | **52.4%** | **2.09** | 29.0% | 2.68 |

**Approach B** stayed at noise floor the entire run:

| step | val_redshift_acc | val_loss_redshift | val_spectrum_acc | val_loss_spectrum |
|---|---|---|---|---|
| 500 | 0.7% | 4.92 | 0.0% | 6.62 |
| 1500 | 0.3% | 4.77 | 11.2% | 3.70 |
| 3000 | 1.4% | 4.75 | 20.4% | 3.21 |
| 4500 | 2.3% | 4.66 | 24.0% | 3.00 |
| 5500 | 1.3% | 4.75 | 25.8% | 2.91 |
| 6500 | 0.25% | 4.65 | 26.5% | 2.86 |
| 8000 | 1.1% | 4.56 | 27.3% | 2.81 |
| 9500 | 1.8% | 4.60 | 28.2% | 2.76 |
| 10000 | 1.2% | 4.53 | 28.3% | 2.74 |

B's `loss_redshift` dropped only 4.92 → 4.53 in 10000 steps (A's dropped 5.10 → 2.09). The encoder is not learning to encode redshift into its hidden state from spectrum features.

### Interpretation: the project's thesis answered

The project's hypothesis (from the assignment, addressing the AION-1 critique): **forcing reconstruction of redshift from spectral context every step** (Approach B) should make redshift an organizing principle of the encoder representation. The result, with our 100M model + 10000 training steps + frozen pretrained tokenizer + 395k spectra:

**B does not work.** When the encoder doesn't see the redshift token directly, the encoder simply leaves redshift unlearned. The decoder, given no redshift signal in cross-attention context, cannot recover the value, and the position-0 loss term stays near `log(256)/log(e) ≈ 5.5` (random over 256 bins).

**A trivially succeeds**, but for an uninteresting reason. With the redshift token included in the encoder input, the decoder learns a cross-attention copy pattern that lifts redshift from encoder position 1 to decoder position 0. This is the same trivial-copy phenomenon that inflates `spec_acc` (see Section: "Proof"). It tests neither spectroscopy nor representation learning — only attention-pattern discovery.

### Proof: the unweighted runs' 99% spec_acc was trivial copy

Pre-fix runs without redshift weighting (jobs 52827566 / 52827575) reached:
- A: val_spectrum_acc 97.2% at step 9000
- B: val_spectrum_acc 99.97% at step 9000

These accuracies on held-out unseen galaxies cannot be memorization. The mechanism is structural:

1. The decoder at position `j` (`j ≥ 1`) is predicting spectrum token `s_j`.
2. The encoder is `[SOS, (redshift,) s1, s2, ..., sN, EOS]` — the same `s_j` sits at encoder position `j` (B) or `j+1` (A).
3. Cross-attention is unmasked. The decoder learns one attention head: "from decoder position `j`, attend to encoder position `j` (B) or `j+1` (A), copy that token."
4. This is a positional shift-and-copy pattern, *data-independent*. It works on every galaxy the model has ever seen and every galaxy it will ever see, because it doesn't depend on galaxy identity.

Evidence this is the mechanism:

- The val_spectrum_acc curve from step 5500 → 7000 jumped from 50% to 99% in 1500 steps for B. This is consistent with "the model just discovered the right attention pattern," not "the model spent 1500 steps learning more spectroscopy."
- B reaches 99.97% but A only 97.2% — the offset in B is simpler (shift by 1) than in A (shift by 2 because of the redshift token), so B converges faster and tighter.
- `val_redshift_acc` stayed at random (~2–4%) the entire pre-fix run: position 0 cannot be solved by copy (B has no source; A has a source at position 1 but the 1/274 gradient share is too small to motivate the copy from position 1 vs from the trivial position-1+offset rule the model has already discovered).

Conclusion: **under the current encoder-decoder + teacher-forced + unmasked-cross-attention architecture, `val_spectrum_acc` does not measure spectrum understanding**. It measures whether the cross-attention copy pattern has been discovered. An honest spectrum reconstruction metric requires either encoder masking (BERT-style) or autoregressive evaluation without teacher forcing.

### AION-1 critique revisited

The original pitch: AION-1 treated redshift like any other token, masked occasionally, so redshift never became an organizing principle of the encoder representation. Our project would fix this by *always* masking redshift (Approach B) and forcing the encoder to encode it.

The current result refines the critique. AION-1's failure mode is real, but **always-masking does not by itself make the encoder encode redshift**. The encoder leaves redshift unlearned regardless of how aggressively the loss penalizes the decoder's failure to recover it. The information has to enter the encoder representation through some *constructive* mechanism, not just through the absence of an alternative shortcut.

Candidate mechanisms for making B work (none tested in this phase):

- **Auxiliary redshift head on the encoder.** Pool encoder outputs (mean, max, or CLS-token style) and predict z from the pooled vector with an auxiliary cross-entropy loss. This is closer to what AION-1 itself did with a separate frozen head, but we'd train it jointly to apply gradient pressure on the encoder.
- **Contrastive loss.** Pull together encoder representations of galaxies with similar redshift; push apart those with different redshift. Forces the encoder's geometry to align with z.
- **Larger encoder capacity / longer training.** B's `loss_redshift` was essentially flat after step 4000, so this is the least likely fix. The information bottleneck appears architectural, not capacity-bound.
- **Continuous redshift loss + scalar head** (instead of discrete bin classification). May give cleaner gradient signal than 256-way softmax.

### Implications for Phase 10

The weight=50 fix (Phase 9) is the right value for A and we should keep it as default. B's failure is not a hyperparameter problem; it's a structural one. Phase 10's encoder masking (next) fixes the `spec_acc` honesty problem and gives us a real metric for both A and B. After that, we can decide whether to test one of the B-rescue candidates above as Phase 11.

### Files referenced
- Train metrics for A: `$SCRATCH/deepsrch/checkpoints/approach_a_52840231/metrics.jsonl`
- Train metrics for B: `$SCRATCH/deepsrch/checkpoints/approach_b_52840234/metrics.jsonl`
- Pre-fix unweighted A: `$SCRATCH/deepsrch/checkpoints/approach_a_52827566/metrics.jsonl`
- Pre-fix unweighted B: `$SCRATCH/deepsrch/checkpoints/approach_b_52827575/metrics.jsonl`
- Tokenizer used: `$SCRATCH/deepsrch/checkpoints/tokenizer_v1_52693687/best.pt`
