#!/bin/bash
#SBATCH -A deepsrch_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH -J tok-v2-s1
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#
# Stage 1: Stabilized V2 tokenizer training on 1000 healpix (~2M spectra).
# lr=1e-4, warmup=1000, commitment=0.05, entropy=0.1
# Expected: ~5-8h wallclock at ~1 step/s (I/O bound), 20k steps.
# Success criteria (step 10000): val_recon < 5.0 AND val_quant > 0.01
#
# Submit from repo root:
#   sbatch nersc/pretrain_tokenizer_v2_s1.slurm
#
# Override account: sbatch -A <other_account> nersc/pretrain_tokenizer_v2_s1.slurm

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load "${PYTORCH_MODULE:-pytorch/2.8.0}"

REPO=$(pwd)
RUN_NAME="tokenizer_v2_s1"
SCRATCH_OUT="$SCRATCH/deepsrch"
MANIFEST="/pscratch/sd/j/joe2k/deepsrch/manifests/dr1_1k_scratch.jsonl"
CFS_OUT="/global/cfs/cdirs/deepsrch/joe2k/checkpoints/tokenizer_v2_s1"

echo "[s1] repo=$REPO"
echo "[s1] run_name=$RUN_NAME"
echo "[s1] manifest=$MANIFEST"

srun -n 1 -c 32 -G 1 python "$REPO/nersc/pretrain_tokenizer_v2.py" \
    --manifest "$MANIFEST" \
    --run-name "$RUN_NAME" \
    --scratch-out "$SCRATCH_OUT/checkpoints" \
    --cfs-out "$CFS_OUT" \
    --steps 20000 \
    --batch-size 32 \
    --lr 1e-4 \
    --warmup 1000 \
    --weight-decay 0.05 \
    --grad-clip 1.0 \
    --num-workers 8 \
    --amp \
    --log-every 20 \
    --val-every 500 \
    --save-every 2000 \
    --wandb-mode offline \
    --entropy-weight 0.1 \
    --commitment-weight 0.05 \
    --use-tophat \
    --use-skip \
    --use-cross-attention

echo "[s1] DONE"