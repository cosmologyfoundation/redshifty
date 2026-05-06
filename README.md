# Unimodal Foundation Model for DESI Spectra

A course project for PHYS303/CS686 — Deep Learning & Bayesian Learning (Spring 2026).

## Overview

This repository implements a **unimodal foundation model for astrophysical spectra**, focusing exclusively on DESI (Dark Energy Spectroscopic Instrument) spectra and redshift (`z`). The project directly addresses a key critique of the AION-1 multimodal foundation model: its redshift token was treated identically to spectral tokens and only masked occasionally, preventing redshift from becoming an organizing principle of the learned representation.

Our model inverts AION-1's breadth-for-depth trade-off: **one modality, treated seriously**.

## Key Contributions

- **Approach A**: Joint training with a lightweight redshift predictor MLP head attached to the encoder representation, trained simultaneously with masked spectral token reconstruction.
- **Approach B**: Always-mask the redshift token — the model must reconstruct `z` from spectral context on every training step.
- Continuous **visualization and testing** at every stage.

## Data

- **DESI Data Release 1** (Early Data Release / SV3 "one-percent" survey)
- ~1 million galaxies, stars, and quasars
- 7,081-pixel wavelength grid spanning ~3,600–9,800 Å
- Redshift provided by DESI pipeline

## Quick Start

### Local Smoke Testing (Mac MPS)

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest

# Verify package imports
python -c "import src; print('OK')"
```

### NERSC A100 Training

See `scripts/` for SLURM job scripts. NERSC uses SLURM with specific constraints:
- Must specify `--constraint`, `--qos`, `--account`, `--gpus`
- GPU jobs require `--gpus` or `-G` flag for CUDA visibility

```bash
# Example submission
sbatch scripts/train_approach_a.sh
```

## Repository Structure

```
FoundationModel/
├── data/                  # Data download & caching scripts
├── src/                   # Source code
│   ├── tokenizers/        # Spectrum & redshift tokenizers
│   ├── models/            # Transformer encoder-decoder
│   ├── training/          # Training loops (Approach A & B)
│   ├── evaluation/        # Metrics & benchmarking
│   └── utils/             # Plotting, logging, config
├── tests/                 # pytest suite
├── notebooks/             # Visualization notebooks
├── scripts/               # SLURM/job scripts for NERSC
└── RESEARCH_LOG.md        # Living document of findings
```

## Evaluation

The model will be tested on a held-out benchmark including:
1. **Redshift prediction** accuracy vs. DESI pipeline values
2. **Spectrum reconstruction** of masked spectral regions
3. **Out-of-distribution generalization** to non-DESI spectra

## References

- AION-1 Paper: Parker et al. (2025), *AION-1: Omnimodal Foundation Model for Astronomical Sciences*
- DESI Collaboration et al. (2016, 2024)

## License

MIT
