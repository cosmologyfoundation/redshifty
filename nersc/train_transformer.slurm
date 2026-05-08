#!/bin/bash
#SBATCH -A deepsrch_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH -J xform-train
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#
# Full SpectrumTransformer training (Approach A or B) on DR1 with a
# pretrained tokenizer. shared QOS, 1 A100, 24h budget.
#
# REQUIRED env: TOKENIZER_CKPT  (path to pretrained tokenizer best.pt)
# OPTIONAL env:
#   APPROACH       a or b              (default: a)
#   MAX_HEALPIX    healpix cap         (default: 2000)
#   STEPS          optimizer steps     (default: 100000)
#   BATCH_SIZE     per-step batch      (default: 8)
#   LR             base lr             (default: 2e-4)
#   NUM_WORKERS    DataLoader workers  (default: 4)
#   D_MODEL        hidden size         (default: 768)
#   N_LAYERS       enc/dec layer count (default: 6)
#   N_HEADS        attention heads     (default: 12)
#
# Submit (Approach A):
#   TOKENIZER_CKPT=$SCRATCH/deepsrch/checkpoints/<run>/best.pt \
#       APPROACH=a sbatch nersc/train_transformer.slurm
#
# Submit (Approach B):
#   TOKENIZER_CKPT=$SCRATCH/deepsrch/checkpoints/<run>/best.pt \
#       APPROACH=b sbatch nersc/train_transformer.slurm

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
RUN_NAME="${RUN_NAME:-approach_${APPROACH}_${SLURM_JOB_ID}}"

# ---------- run-scale knobs ----------
SURVEYS="sv3 main"
PROGRAMS="bright dark"
MAX_HEALPIX="${MAX_HEALPIX:-2000}"
STEPS="${STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-2e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
D_MODEL="${D_MODEL:-768}"
N_LAYERS="${N_LAYERS:-6}"
N_HEADS="${N_HEADS:-12}"
# -------------------------------------

MANIFEST="${MANIFEST:-$SCRATCH_OUT/manifests/dr1_${MAX_HEALPIX}.jsonl}"
CFS_OUT="${CFS_OUT:-$REPO/checkpoints/nersc/$RUN_NAME}"

echo "[train] approach=$APPROACH run=$RUN_NAME"
echo "[train] manifest=$MANIFEST  cfs_out=$CFS_OUT"

if [ ! -s "$MANIFEST" ]; then
    echo "[train] building manifest..."
    python "$REPO/nersc/build_dr1_index.py" \
        --root /global/cfs/cdirs/desi/public/dr1 \
        --surveys $SURVEYS --programs $PROGRAMS \
        --max-healpix "$MAX_HEALPIX" \
        --out "$MANIFEST"
fi

srun -n 1 -c 32 -G 1 python "$REPO/nersc/train_transformer.py" \
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
    --amp

echo "[train] done; best/final mirrored to $CFS_OUT"
