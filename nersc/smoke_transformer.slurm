#!/bin/bash
#SBATCH -A deepsrch_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH -J xform-smoke
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#
# Smoke transformer training: 100 steps, 200 spectra, tiny model.
# Validates that the pretrained tokenizer loads and the train loop runs.
#
# REQUIRED env: TOKENIZER_CKPT (path to pretrained tokenizer best.pt)
# OPTIONAL env: APPROACH (default: a)
#
# Submit:
#   TOKENIZER_CKPT=$SCRATCH/deepsrch/checkpoints/tokenizer_v1_52693687/best.pt \
#       sbatch nersc/smoke_transformer.slurm

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load "${PYTORCH_MODULE:-pytorch/2.8.0}"

if [ -z "${TOKENIZER_CKPT:-}" ]; then
    echo "ERROR: TOKENIZER_CKPT env var is required"
    exit 1
fi

REPO=$(pwd)
APPROACH="${APPROACH:-a}"
RUN_NAME="xform_smoke_${APPROACH}_${SLURM_JOB_ID}"
SCRATCH_OUT="$SCRATCH/deepsrch"
MANIFEST="$SCRATCH_OUT/manifests/dr1_smoke.jsonl"

if [ ! -s "$MANIFEST" ]; then
    echo "[smoke] building tiny manifest..."
    python "$REPO/nersc/build_dr1_index.py" \
        --root /global/cfs/cdirs/desi/public/dr1 \
        --surveys sv3 --programs bright \
        --max-healpix 5 \
        --out "$MANIFEST"
fi

srun -n 1 -c 32 -G 1 python "$REPO/nersc/train_transformer.py" \
    --manifest "$MANIFEST" \
    --tokenizer-ckpt "$TOKENIZER_CKPT" \
    --approach "$APPROACH" \
    --run-name "$RUN_NAME" \
    --scratch-out "$SCRATCH_OUT" \
    --smoke

echo "[smoke] OK"
