#!/bin/bash
#SBATCH -A deepsrch_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH -J tok-pretrain
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#
# Full tokenizer pretrain on a single A100 in shared QOS.
# `shared` lets us request just 1 GPU (vs a whole 4-GPU node) so we burn
# 1/4 the allocation hours. Max 48h wallclock; we use 24h.
#
# Configure SURVEYS / PROGRAMS / TARGET_SPECTRA below for the run scale
# you want.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load "${PYTORCH_MODULE:-pytorch/2.3.1}"

REPO=$(pwd)
SCRATCH_OUT="$SCRATCH/deepsrch"
RUN_NAME="${RUN_NAME:-tokenizer_v1_${SLURM_JOB_ID}}"

# ---------- run-scale knobs ----------
SURVEYS="sv3 main"
PROGRAMS="bright dark"
MAX_HEALPIX="${MAX_HEALPIX:-2000}"   # ~ a few hundred thousand spectra
STEPS="${STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-3e-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
# -------------------------------------

MANIFEST="$SCRATCH_OUT/manifests/dr1_${SLURM_JOB_ID}.jsonl"
CFS_OUT="${CFS_OUT:-$REPO/checkpoints/nersc/$RUN_NAME}"

echo "[pretrain] run_name=$RUN_NAME"
echo "[pretrain] manifest=$MANIFEST  (max_healpix=$MAX_HEALPIX)"
echo "[pretrain] cfs_out=$CFS_OUT"

if [ ! -s "$MANIFEST" ]; then
    echo "[pretrain] building manifest..."
    python "$REPO/nersc/build_dr1_index.py" \
        --root /global/cfs/cdirs/desi/public/dr1 \
        --surveys $SURVEYS \
        --programs $PROGRAMS \
        --max-healpix "$MAX_HEALPIX" \
        --out "$MANIFEST"
fi

srun -n 1 -c 32 -G 1 python "$REPO/nersc/pretrain_tokenizer.py" \
    --manifest "$MANIFEST" \
    --run-name "$RUN_NAME" \
    --scratch-out "$SCRATCH_OUT/checkpoints" \
    --cfs-out "$CFS_OUT" \
    --steps "$STEPS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --num-workers "$NUM_WORKERS" \
    --amp

echo "[pretrain] done; best.pt mirrored to $CFS_OUT"
