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
#SBATCH -J xform-ddp
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#
# 4-GPU DDP training of SpectrumTransformer on a full Perlmutter node.
# Regular QOS, 12h budget.  srun spawns 4 Python procs, each bound to
# 1 GPU via LOCAL_RANK.
#
# REQUIRED env: TOKENIZER_CKPT  (path to pretrained tokenizer best.pt)
# OPTIONAL env:
#   APPROACH       a or b              (default: a)
#   MAX_HEALPIX    healpix cap         (default: 10000)
#   STEPS          optimizer steps     (default: 100000)
#   BATCH_SIZE     per-GPU batch       (default: 32)
#   LR             base lr             (default: 8e-4, linear-scaled 4x)
#   NUM_WORKERS    DataLoader workers  (default: 8)
#   D_MODEL        hidden size         (default: 768)
#   N_LAYERS       enc/dec layer count (default: 6)
#   N_HEADS        attention heads     (default: 12)
#
# Submit:
#   TOKENIZER_CKPT=$SCRATCH/deepsrch/checkpoints/<run>/best.pt \
#       MANIFEST=$SCRATCH/deepsrch/manifests/dr1_10k.jsonl \
#       APPROACH=a sbatch nersc/train_transformer_ddp.slurm

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load "${PYTORCH_MODULE:-pytorch/2.8.0}"

if [ -z "${TOKENIZER_CKPT:-}" ]; then
    echo "ERROR: TOKENIZER_CKPT env var is required"
    exit 1
fi

REPO=$(pwd)
APPROACH="${APPROACH:-a}"
SCRATCH_OUT="$SCRATCH/deepsrch"
RUN_NAME="${RUN_NAME:-approach_${APPROACH}_ddp4_${SLURM_JOB_ID}}"

# ---------- run-scale knobs ----------
SURVEYS="sv3 main"
PROGRAMS="bright dark"
MAX_HEALPIX="${MAX_HEALPIX:-10000}"
STEPS="${STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-8e-4}"
NUM_WORKERS="${NUM_WORKERS:-12}"
D_MODEL="${D_MODEL:-768}"
N_LAYERS="${N_LAYERS:-6}"
N_HEADS="${N_HEADS:-12}"
REDSHIFT_LOSS_WEIGHT="${REDSHIFT_LOSS_WEIGHT:-10.0}"
ENCODER_MASK_RATIO="${ENCODER_MASK_RATIO:-0.50}"
HEALPIX_HOLDOUT_FRAC="${HEALPIX_HOLDOUT_FRAC:-0.05}"
AR_EVAL_BATCHES="${AR_EVAL_BATCHES:-8}"
WANDB_MODE_ARG="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-redshifty}"
# -------------------------------------

MANIFEST="${MANIFEST:-$SCRATCH_OUT/manifests/dr1_${MAX_HEALPIX}.jsonl}"
DEFAULT_CFS_BASE="/global/cfs/cdirs/deepsrch/$USER/checkpoints"
CFS_OUT="${CFS_OUT:-$DEFAULT_CFS_BASE/$RUN_NAME}"

echo "[train-ddp] approach=$APPROACH run=$RUN_NAME"
echo "[train-ddp] manifest=$MANIFEST  cfs_out=$CFS_OUT"
echo "[train-ddp] 4 GPUs, per-gpu batch=$BATCH_SIZE, effective batch=$((BATCH_SIZE * 4)), lr=$LR"

if [ ! -s "$MANIFEST" ]; then
    echo "[train-ddp] building manifest..."
    python "$REPO/nersc/build_dr1_index.py" \
        --root /global/cfs/cdirs/desi/public/dr1 \
        --surveys $SURVEYS --programs $PROGRAMS \
        --max-healpix "$MAX_HEALPIX" \
        --out "$MANIFEST"
fi

srun python "$REPO/nersc/train_transformer.py" \
    --manifest "$MANIFEST" \
    --tokenizer-ckpt "$TOKENIZER_CKPT" \
    --approach "$APPROACH" \
    --run-name "$RUN_NAME" \
    --scratch-out "$SCRATCH_OUT" \
    --cfs-out "$CFS_OUT" \
    --steps "$STEPS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --num-workers "$NUM_WORKERS" \
    --d-model "$D_MODEL" \
    --n-encoder-layers "$N_LAYERS" \
    --n-decoder-layers "$N_LAYERS" \
    --n-heads "$N_HEADS" \
    --redshift-loss-weight "$REDSHIFT_LOSS_WEIGHT" \
    --encoder-mask-ratio "$ENCODER_MASK_RATIO" \
    --healpix-holdout-frac "$HEALPIX_HOLDOUT_FRAC" \
    --ar-eval-batches "$AR_EVAL_BATCHES" \
    --wandb-mode "$WANDB_MODE_ARG" \
    --wandb-project "$WANDB_PROJECT" \
    --amp

echo "[train-ddp] done; best/final mirrored to $CFS_OUT"
