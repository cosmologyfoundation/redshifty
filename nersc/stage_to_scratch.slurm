#!/bin/bash
#SBATCH -A deepsrch
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH -J stage-dr1
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#
# Stage a DR1 manifest from $CFS to $SCRATCH so training jobs read from
# the high-perf Lustre filesystem instead of CFS.
#
# Note: this is a CPU-only job (no GPU needed), so we drop the _g suffix
# and use $CFS-readable CPU allocation. ~50 GB at ~200 MB/s ≈ 5 minutes;
# 1h time limit is conservative.
#
# Required env:
#   SRC_MANIFEST     JSONL from build_dr1_index.py (CFS paths)
#   DST_MANIFEST     where to write the rewritten JSONL (defaults under $SCRATCH)
#
# Optional env:
#   DST_ROOT         directory under which files are mirrored (default $SCRATCH/deepsrch/dr1_staged)
#   MAX_FILES        cap on number of healpix dirs to stage
#
# Example:
#   SRC_MANIFEST=$SCRATCH/deepsrch/manifests/dr1_200.jsonl \
#       sbatch nersc/stage_to_scratch.slurm

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load "${PYTORCH_MODULE:-pytorch/2.8.0}"

if [ -z "${SRC_MANIFEST:-}" ]; then
    echo "ERROR: SRC_MANIFEST env var is required" >&2
    exit 1
fi

REPO=$(pwd)
DST_ROOT="${DST_ROOT:-$SCRATCH/deepsrch/dr1_staged}"
DST_MANIFEST="${DST_MANIFEST:-${SRC_MANIFEST%.jsonl}_scratch.jsonl}"
MAX_FILES_ARG=""
if [ -n "${MAX_FILES:-}" ]; then
    MAX_FILES_ARG="--max-files ${MAX_FILES}"
fi

echo "[stage] SRC_MANIFEST = $SRC_MANIFEST"
echo "[stage] DST_ROOT     = $DST_ROOT"
echo "[stage] DST_MANIFEST = $DST_MANIFEST"

python "$REPO/nersc/stage_to_scratch.py" \
    --src-manifest "$SRC_MANIFEST" \
    --dst-root "$DST_ROOT" \
    --dst-manifest "$DST_MANIFEST" \
    $MAX_FILES_ARG

echo "[stage] OK. To use in training:"
echo "    MANIFEST=$DST_MANIFEST sbatch nersc/train_transformer.slurm"
