#!/usr/bin/env bash
# run_simulator.sh — launch NVFlare simulator with SparseFedMoE job defaults.
#
# Usage: scripts/run_simulator.sh <job_dir> [-n NUM_CLIENTS] [-t THREADS] [-w WORKSPACE]
#
# Defaults chosen for laptop smoke runs:
#   - 2 clients, 1 thread, workspace in /tmp
#   - SPARSEFEDMOE_DUMMY_MODEL=1 unless caller overrides
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <job_dir> [-n N] [-t T] [-w WORKSPACE]" >&2
  exit 1
fi

JOB_DIR="$1"; shift
NUM_CLIENTS=7
THREADS=1
WORKSPACE="/tmp/nvflare/sparsefedmoe"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n) NUM_CLIENTS="$2"; shift 2 ;;
    -t) THREADS="$2"; shift 2 ;;
    -w) WORKSPACE="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

export SPARSEFEDMOE_DUMMY_MODEL="${SPARSEFEDMOE_DUMMY_MODEL:-1}"
# Resolve to an absolute path: the simulator changes cwd to each site's
# workspace under $WORKSPACE, so a relative ./data wouldn't find the project's
# partitions.
_DEFAULT_DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data"
export SPARSEFEDMOE_DATA="${SPARSEFEDMOE_DATA:-$_DEFAULT_DATA_DIR}"
export SPARSEFEDMOE_LOCAL_EPOCHS="${SPARSEFEDMOE_LOCAL_EPOCHS:-1}"
export SPARSEFEDMOE_BATCH_SIZE="${SPARSEFEDMOE_BATCH_SIZE:-2}"
export SPARSEFEDMOE_MAX_SEQ_LEN="${SPARSEFEDMOE_MAX_SEQ_LEN:-32}"

echo "─── NVFlare simulator ───"
echo "  job:       $JOB_DIR"
echo "  clients:   $NUM_CLIENTS"
echo "  threads:   $THREADS"
echo "  workspace: $WORKSPACE"
echo "  dummy:     $SPARSEFEDMOE_DUMMY_MODEL"
mkdir -p "$WORKSPACE"

nvflare simulator \
  "$JOB_DIR" \
  -w "$WORKSPACE" \
  -n "$NUM_CLIENTS" \
  -t "$THREADS"
