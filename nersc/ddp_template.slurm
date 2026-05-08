#!/bin/bash
#SBATCH -A deepsrch_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpu-bind=none
#SBATCH --time=12:00:00
#SBATCH -J tok-ddp
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#
# TEMPLATE for multi-GPU DDP runs (full Perlmutter node, 4x A100).
# pretrain_tokenizer.py is currently single-GPU; this script is here as
# a ready-to-fill scaffold once the trainer is promoted to DDP.
#
# Required code change before this script will work:
#   - Wrap model in torch.nn.parallel.DistributedDataParallel
#   - Use torch.distributed.init_process_group("nccl") with env init
#   - Replace DataLoader's shuffle=True with a DistributedSampler
#
# The env-var translation (RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR,
# MASTER_PORT) is already handled by export_ddp_vars.sh.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load "${PYTORCH_MODULE:-pytorch/2.3.1}"

REPO=$(pwd)
SCRATCH_OUT="$SCRATCH/deepsrch"
RUN_NAME="${RUN_NAME:-tokenizer_ddp_${SLURM_JOB_ID}}"
MANIFEST="${MANIFEST:-$SCRATCH_OUT/manifests/dr1_full.jsonl}"

srun -l bash -c "
    source $REPO/nersc/export_ddp_vars.sh
    python $REPO/nersc/pretrain_tokenizer.py \
        --manifest $MANIFEST \
        --run-name $RUN_NAME \
        --scratch-out $SCRATCH_OUT/checkpoints \
        --steps 200000 \
        --batch-size 32 \
        --num-workers 8 \
        --amp
"
