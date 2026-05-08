#!/usr/bin/env bash
# One-time environment setup on Perlmutter.
#
# Run from a login node:
#     bash nersc/setup_env.sh
#
# What this does:
#   1. Loads NERSC's pytorch module (CUDA + PyTorch + Python all preconfigured).
#   2. Adds project python deps via `pip install --user` so they layer onto
#      the module without breaking it. NERSC sets PYTHONUSERBASE per module
#      so these installs stay scoped.
#   3. Creates working dirs on $SCRATCH for checkpoints/logs.
#
# Re-run this if you switch the pytorch module version.

set -euo pipefail

# ---------- Edit this if a newer pytorch module is preferred ----------
PYTORCH_MODULE="${PYTORCH_MODULE:-pytorch/2.8.0}"
# ----------------------------------------------------------------------

echo "[1/3] module load $PYTORCH_MODULE"
module load "$PYTORCH_MODULE"

echo "[2/3] pip install project deps"
# Note: --user installs into $PYTHONUSERBASE which the pytorch module sets
# to a per-module location. astropy + tqdm are the only things missing
# from the base module.
python -m pip install --user --no-cache-dir \
    "astropy>=6.0" \
    "fitsio>=1.2" \
    "tqdm>=4.66"

echo "[3/3] create scratch dirs"
mkdir -p "$SCRATCH/deepsrch/checkpoints"
mkdir -p "$SCRATCH/deepsrch/manifests"
mkdir -p "$SCRATCH/deepsrch/logs"

echo
echo "Done."
echo "  PyTorch module : $PYTORCH_MODULE"
echo "  Scratch root   : $SCRATCH/deepsrch"
echo
echo "Next: build the DR1 manifest (see nersc/README.md)."
