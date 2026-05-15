#!/bin/bash
#SBATCH -A deepsrch_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus=4
#SBATCH --gpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH -J fm_v3_option_a
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#
# 4-GPU DDP training of SpectrumTransformer with Option A dual cross-attention.
# Regular QOS, 12h budget. torchrun spawns 4 Python procs, each bound to 1 GPU.
#
# REQUIRED env: TOKENIZER_CKPT, MANIFEST
# OPTIONAL env:
#   RUN_NAME       run name (default: fm_v3_option_a_${SLURM_JOB_ID})
#   STEPS          optimizer steps (default: 50000)
#   BATCH_SIZE     per-GPU batch (default: 32)
#   LR             base lr (default: 2e-4)
#   NUM_WORKERS    DataLoader workers (default: 8)
#   REDSHIFT_LOSS_WEIGHT   (default: 50)
#   ENCODER_MASK_RATIO     (default: 0.50)
#   DECODER_CORRUPT_RATIO  (default: 0.05)
#   HEALPIX_HOLDOUT_FRAC   (default: 0.10)
#
# Submit:
#   TOKENIZER_CKPT=$SCRATCH/deepsrch/checkpoints/tokenizer_v2_s1/best.pt \
#       MANIFEST=$SCRATCH/deepsrch/manifests/dr1_10k_scratch.jsonl \
#       sbatch nersc/train_transformer_option_a.slurm

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load "${PYTORCH_MODULE:-pytorch/2.8.0}"

if [ -z "${TOKENIZER_CKPT:-}" ]; then
    echo "ERROR: TOKENIZER_CKPT env var is required"
    exit 1
fi

REPO=$(pwd)
RUN_NAME="${RUN_NAME:-fm_v3_option_a_${SLURM_JOB_ID}}"
SCRATCH_OUT="$SCRATCH/deepsrch"
STEPS="${STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-2e-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
REDSHIFT_LOSS_WEIGHT="${REDSHIFT_LOSS_WEIGHT:-50}"
ENCODER_MASK_RATIO="${ENCODER_MASK_RATIO:-0.50}"
DECODER_CORRUPT_RATIO="${DECODER_CORRUPT_RATIO:-0.05}"
HEALPIX_HOLDOUT_FRAC="${HEALPIX_HOLDOUT_FRAC:-0.10}"
WANDB_MODE_ARG="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-redshifty}"

MANIFEST="${MANIFEST:-$SCRATCH_OUT/manifests/dr1_10k_scratch.jsonl}"
DEFAULT_CFS_BASE="/global/cfs/cdirs/deepsrch/$USER/checkpoints"
CFS_OUT="${CFS_OUT:-$DEFAULT_CFS_BASE/$RUN_NAME}"

echo "[option_a] run=$RUN_NAME"
echo "[option_a] manifest=$MANIFEST  cfs_out=$CFS_OUT"
echo "[option_a] 4 GPUs, per-gpu batch=$BATCH_SIZE, effective batch=$((BATCH_SIZE * 4)), lr=$LR"

# NCCL knobs for Slingshot-11
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

python -m torch.distributed.run --nproc_per_node=4 \
    "$REPO/nersc/train_transformer.py" \
    --manifest "$MANIFEST" \
    --tokenizer-ckpt "$TOKENIZER_CKPT" \
    --tokenizer-kind v2 \
    --redshift-levels 1024 \
    --approach a \
    --run-name "$RUN_NAME" \
    --scratch-out "$SCRATCH_OUT" \
    --cfs-out "$CFS_OUT" \
    --steps "$STEPS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --num-workers "$NUM_WORKERS" \
    --redshift-loss-weight "$REDSHIFT_LOSS_WEIGHT" \
    --encoder-mask-ratio "$ENCODER_MASK_RATIO" \
    --decoder-corrupt-ratio "$DECODER_CORRUPT_RATIO" \
    --healpix-holdout-frac "$HEALPIX_HOLDOUT_FRAC" \
    --wandb-mode "$WANDB_MODE_ARG" \
    --wandb-project "$WANDB_PROJECT" \
    --amp

echo "[option_a] done; best/final mirrored to $CFS_OUT"