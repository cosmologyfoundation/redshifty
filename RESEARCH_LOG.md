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

1. Phase 1: Build minimal smoke-test data pipeline
2. Phase 2: Adapt spectrum tokenizer from AION
3. Phase 3: Redshift scalar tokenizer
4. Phase 4: Transformer backbone
5. Phase 5: Approach A training
6. Phase 6: Approach B training
7. Phase 7: Evaluation & comparison
8. Phase 8: NERSC full-scale training
9. Phase 9: OOD generalization prep

---
