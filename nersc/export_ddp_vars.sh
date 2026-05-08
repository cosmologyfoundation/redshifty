#!/usr/bin/env bash
# Translate SLURM env vars -> torch.distributed env vars.
# Source this from inside an `srun` invocation when using DDP.
#
#     srun -l bash -c "source nersc/export_ddp_vars.sh; python script.py ..."
#
# Cribbed from NERSC's nersc-dl-multigpu/export_DDP_vars.sh.

export RANK=${SLURM_PROCID}
export LOCAL_RANK=${SLURM_LOCALID}
export WORLD_SIZE=${SLURM_NTASKS}
# Pick a free-ish port deterministic per job
export MASTER_PORT=$(( 29500 + (SLURM_JOB_ID % 1000) ))
# Master is the first node in the allocation
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)

# NCCL knobs that NERSC recommends for Slingshot-11
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
