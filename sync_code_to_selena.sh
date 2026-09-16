#!/bin/bash

set -euo pipefail

usage() {
    echo "usage: bash sync_code_to_selena.sh [--dry-run]" >&2
}

RSYNC_OPTIONS=()
DRY_RUN=false
while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run) RSYNC_OPTIONS+=(--dry-run); DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage; echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
PROJECT_NAME="$(basename "$PROJECT_ROOT")"
NNI_FILE="${TIME_NNI_FILE:-$HOME/codes/.secrets/nni}"
if [ ! -f "$NNI_FILE" ]; then
    echo "ERROR: missing $NNI_FILE" >&2
    exit 1
fi

NNI="$(sed -n '1p' "$NNI_FILE" | tr -d '[:space:]')"
nni="${NNI,,}"
if [[ ! "$nni" =~ ^[a-z][a-z0-9_-]*$ ]]; then
    echo "ERROR: $NNI_FILE must contain one valid NNI" >&2
    exit 1
fi

SELENA_HOST="${TIME_SELENA_HOST:-$nni@selena.hpc.edf.fr}"
DESTINATION="${TIME_SELENA_CODE_ROOT:-$SELENA_HOST:~/codes/$PROJECT_NAME/}"
SCRATCH_STORAGE_ROOT="${TIME_SELENA_STORAGE_ROOT:-/scratch/users/$nni}"
SCRATCH_PROJECT_ROOT="${TIME_SELENA_SCRATCH_ROOT:-$SCRATCH_STORAGE_ROOT/codes/$PROJECT_NAME}"

echo "Synchronizing $PROJECT_NAME code from DGX to Selena..."
rsync -rlptz --delete-delay --itemize-changes --partial --info=progress2 \
    "${RSYNC_OPTIONS[@]}" \
    --exclude='.git/' \
    --exclude='.env' \
    --exclude='.venv' \
    --exclude='.secrets/' \
    --exclude='pyproject.toml' \
    --exclude='uv.lock' \
    --exclude='AGENTS.md' \
    --exclude='FUTURE_WORK.md' \
    --exclude='PENDING_UPDATES.md' \
    --exclude='CLUSTER_STATUS.txt' \
    --exclude='docs/INTERNAL_WORKFLOW.md' \
    --exclude='docs/IMPROVEMENTS.md' \
    --exclude='datasets/' \
    --exclude='weights/' \
    --exclude='archive/' \
    --exclude='outputs/' \
    --exclude='logs/' \
    "$PROJECT_ROOT/" \
    "$DESTINATION"

if [ "$DRY_RUN" = true ]; then
    echo "PREVIEW: no files were transferred or deleted."
    exit 0
fi

ssh "$SELENA_HOST" \
    "mkdir -p '$SCRATCH_STORAGE_ROOT/datasets' '$SCRATCH_STORAGE_ROOT/weights' '$SCRATCH_STORAGE_ROOT/venvs' '$SCRATCH_PROJECT_ROOT/outputs' '$SCRATCH_PROJECT_ROOT/logs'"

echo "SUCCESS: Selena's $PROJECT_NAME code matches DGX."
echo "Preserved on Selena: .venv, pyproject.toml, and uv.lock."
echo "Selena datasets: $SCRATCH_STORAGE_ROOT/datasets"
echo "Selena weights: $SCRATCH_STORAGE_ROOT/weights"
echo "Selena uv environments: $SCRATCH_STORAGE_ROOT/venvs"
echo "Selena outputs: $SCRATCH_PROJECT_ROOT/outputs"
echo "Selena logs: $SCRATCH_PROJECT_ROOT/logs"
