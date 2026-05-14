#!/bin/bash
#SBATCH -A deepsrch_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH -J tok-v2-smoke
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#
# Smoke test: pretrain V2 spectrum tokenizer for ~50 steps on 1000-healpix manifest.
# Validates: env loads, V2 model init, manifest reads, FITS open from SCRATCH,
# GPU step runs, AMP enabled, checkpoint writes. ~3-5 min wallclock.
#
# Submit from repo root:
#   sbatch nersc/smoke_tokenizer_v2.slurm
#
# Override account at submit time: sbatch -A <other_account> nersc/smoke_tokenizer_v2.slurm

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load "${PYTORCH_MODULE:-pytorch/2.8.0}"

REPO=$(pwd)
RUN_NAME="tokv2_smoke_${SLURM_JOB_ID}"
SCRATCH_OUT="$SCRATCH/deepsrch"
MANIFEST="/pscratch/sd/j/joe2k/deepsrch/manifests/dr1_1k_scratch.jsonl"

echo "[smoke] repo=$REPO"
echo "[smoke] run_name=$RUN_NAME"
echo "[smoke] manifest=$MANIFEST"

# Smoke training: 50 steps, 200 spectra cap, AMP on
srun -n 1 -c 32 -G 1 python "$REPO/nersc/pretrain_tokenizer_v2.py" \
    --manifest "$MANIFEST" \
    --run-name "$RUN_NAME" \
    --scratch-out "$SCRATCH_OUT/checkpoints" \
    --steps 50 \
    --max-spectra 200 \
    --batch-size 4 \
    --val-every 25 \
    --save-every 50 \
    --log-every 5 \
    --amp \
    --num-workers 0 \
    --wandb-mode offline \
    --entropy-weight 0.1 \
    --commitment-weight 0.05 \
    --use-tophat \
    --use-skip \
    --use-cross-attention

echo "[smoke] OK"