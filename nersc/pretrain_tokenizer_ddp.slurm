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
#SBATCH -J tok-ddp
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#
# 4-GPU DDP pretraining of the spectrum tokenizer on a full Perlmutter node.
# Regular QOS, 12h budget.
#
# Submit:
#   MANIFEST=$SCRATCH/deepsrch/manifests/dr1_10k.jsonl \
#       sbatch nersc/pretrain_tokenizer_ddp.slurm

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load "${PYTORCH_MODULE:-pytorch/2.8.0}"

REPO=$(pwd)
SCRATCH_OUT="$SCRATCH/deepsrch"
RUN_NAME="${RUN_NAME:-tokenizer_ddp4_${SLURM_JOB_ID}}"

# ---------- run-scale knobs ----------
SURVEYS="sv3 main"
PROGRAMS="bright dark"
MAX_HEALPIX="${MAX_HEALPIX:-10000}"
STEPS="${STEPS:-200000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1.2e-3}"
NUM_WORKERS="${NUM_WORKERS:-12}"
# -------------------------------------

MANIFEST="${MANIFEST:-$SCRATCH_OUT/manifests/dr1_${MAX_HEALPIX}.jsonl}"
DEFAULT_CFS_BASE="/global/cfs/cdirs/deepsrch/$USER/checkpoints"
CFS_OUT="${CFS_OUT:-$DEFAULT_CFS_BASE/$RUN_NAME}"

echo "[tok-ddp] run_name=$RUN_NAME"
echo "[tok-ddp] manifest=$MANIFEST  (max_healpix=$MAX_HEALPIX)"
echo "[tok-ddp] cfs_out=$CFS_OUT"
echo "[tok-ddp] 4 GPUs, per-gpu batch=$BATCH_SIZE, effective batch=$((BATCH_SIZE * 4)), lr=$LR"

if [ ! -s "$MANIFEST" ]; then
    echo "[tok-ddp] building manifest..."
    python "$REPO/nersc/build_dr1_index.py" \
        --root /global/cfs/cdirs/desi/public/dr1 \
        --surveys $SURVEYS \
        --programs $PROGRAMS \
        --max-healpix "$MAX_HEALPIX" \
        --out "$MANIFEST"
fi

# Translate SLURM vars -> PyTorch DDP env vars
export RANK=${SLURM_PROCID}
export LOCAL_RANK=${SLURM_LOCALID}
export WORLD_SIZE=${SLURM_NTASKS}
export MASTER_PORT=$(( 29500 + (SLURM_JOB_ID % 1000) ))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)

# NCCL knobs for Slingshot-11
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

srun python "$REPO/nersc/pretrain_tokenizer.py" \
    --manifest "$MANIFEST" \
    --run-name "$RUN_NAME" \
    --scratch-out "$SCRATCH_OUT/checkpoints" \
    --cfs-out "$CFS_OUT" \
    --steps "$STEPS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --num-workers "$NUM_WORKERS" \
    --amp

echo "[tok-ddp] done; best.pt mirrored to $CFS_OUT"
