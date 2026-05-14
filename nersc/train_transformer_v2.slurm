#!/bin/bash
#SBATCH -A deepsrch_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH -J xform-v2-train
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#
# Full SpectrumTransformer V2 training (Approach A or B) on DR1 with
# SpectrumTokenizerV2 (val_recon=0.157). Uses redshift tokenizer V2
# (1024 FSQ levels). shared QOS, 1 A100, 24h budget.
#
# REQUIRED env: TOKENIZER_CKPT  (path to pretrained SpectrumTokenizerV2 best.pt)
# OPTIONAL env:
#   APPROACH       a or b              (default: a)
#   MAX_HEALPIX    healpix cap         (default: 2000)
#   STEPS          optimizer steps     (default: 100000)
#   BATCH_SIZE     per-step batch      (default: 32)
#   LR             base lr             (default: 4e-4)
#   NUM_WORKERS    DataLoader workers  (default: 4)
#   D_MODEL        hidden size         (default: 768)
#   N_LAYERS       enc/dec layer count (default: 6)
#   N_HEADS        attention heads     (default: 12)
#   REDSHIFT_LOSS_WEIGHT  weight on position-0 loss (default: 50)
#   ENCODER_MASK_RATIO    BERT-style spectrum masking (default: 0.50)
#
# Submit (Approach A):
#   TOKENIZER_CKPT=/global/cfs/cdirs/deepsrch/joe2k/checkpoints/tokenizer_v2_s1_interactive/best.pt \
#       APPROACH=a BATCH_SIZE=32 LR=4e-4 REDSHIFT_LOSS_WEIGHT=50 ENCODER_MASK_RATIO=0.50 \
#       sbatch nersc/train_transformer_v2.slurm
#
# Submit (Approach B):
#   TOKENIZER_CKPT=/global/cfs/cdirs/deepsrch/joe2k/checkpoints/tokenizer_v2_s1_interactive/best.pt \
#       APPROACH=b BATCH_SIZE=32 LR=4e-4 REDSHIFT_LOSS_WEIGHT=50 ENCODER_MASK_RATIO=0.50 \
#       sbatch nersc/train_transformer_v2.slurm

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
RUN_NAME="${RUN_NAME:-approach_${APPROACH}_v2_${SLURM_JOB_ID}}"

# ---------- run-scale knobs ----------
SURVEYS="sv3 main"
PROGRAMS="bright dark"
MAX_HEALPIX="${MAX_HEALPIX:-2000}"
STEPS="${STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-4e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
D_MODEL="${D_MODEL:-768}"
N_LAYERS="${N_LAYERS:-6}"
N_HEADS="${N_HEADS:-12}"
REDSHIFT_LOSS_WEIGHT="${REDSHIFT_LOSS_WEIGHT:-50.0}"
ENCODER_MASK_RATIO="${ENCODER_MASK_RATIO:-0.50}"
HEALPIX_HOLDOUT_FRAC="${HEALPIX_HOLDOUT_FRAC:-0.05}"
AR_EVAL_BATCHES="${AR_EVAL_BATCHES:-4}"
WANDB_MODE_ARG="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-redshifty}"
# -------------------------------------

MANIFEST="${MANIFEST:-$SCRATCH_OUT/manifests/dr1_${MAX_HEALPIX}.jsonl}"
DEFAULT_CFS_BASE="/global/cfs/cdirs/deepsrch/$USER/checkpoints"
CFS_OUT="${CFS_OUT:-$DEFAULT_CFS_BASE/$RUN_NAME}"

echo "[train_v2] approach=$APPROACH run=$RUN_NAME"
echo "[train_v2] manifest=$MANIFEST  cfs_out=$CFS_OUT"

if [ ! -s "$MANIFEST" ]; then
    echo "[train_v2] building manifest..."
    python "$REPO/nersc/build_dr1_index.py" \
        --root /global/cfs/cdirs/desi/public/dr1 \
        --surveys $SURVEYS --programs $PROGRAMS \
        --max-healpix "$MAX_HEALPIX" \
        --out "$MANIFEST"
fi

srun -n 1 -c 32 -G 1 python "$REPO/nersc/train_transformer.py" \
    --manifest "$MANIFEST" \
    --tokenizer-ckpt "$TOKENIZER_CKPT" \
    --tokenizer-kind v2 \
    --redshift-levels 1024 \
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

echo "[train_v2] done; best/final mirrored to $CFS_OUT"