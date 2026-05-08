#!/bin/bash
#SBATCH -A deepsrch_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH -J tok-smoke
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#
# Smoke test: pretrain spectrum tokenizer for ~50 steps on a tiny manifest.
# Validates: env loads, manifest reads, FITS open from CFS, GPU step runs,
# checkpoint writes to $SCRATCH. ~few-minute wallclock.
#
# Submit from the repo root:
#     sbatch nersc/smoke_tokenizer.slurm
#
# If `--account=deepsrch_g` is wrong for your allocation, override at
# submit time:  sbatch -A <other_account> nersc/smoke_tokenizer.slurm

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load "${PYTORCH_MODULE:-pytorch/2.3.1}"

REPO=$(pwd)
RUN_NAME="tok_smoke_${SLURM_JOB_ID}"
SCRATCH_OUT="$SCRATCH/deepsrch"
MANIFEST="$SCRATCH_OUT/manifests/dr1_smoke.jsonl"

echo "[smoke] repo=$REPO"
echo "[smoke] run_name=$RUN_NAME"
echo "[smoke] manifest=$MANIFEST"

# Build a tiny manifest if it isn't there (5 healpix files; ~few hundred spectra).
if [ ! -s "$MANIFEST" ]; then
    echo "[smoke] building tiny manifest..."
    python "$REPO/nersc/build_dr1_index.py" \
        --root /global/cfs/cdirs/desi/public/dr1 \
        --surveys sv3 \
        --programs bright \
        --max-healpix 5 \
        --out "$MANIFEST"
fi

# Smoke training (50 steps, 200 spectra cap, no AMP)
srun -n 1 -c 32 -G 1 python "$REPO/nersc/pretrain_tokenizer.py" \
    --manifest "$MANIFEST" \
    --run-name "$RUN_NAME" \
    --scratch-out "$SCRATCH_OUT/checkpoints" \
    --smoke

echo "[smoke] OK"
