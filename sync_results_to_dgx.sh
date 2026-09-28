#!/bin/bash

set -euo pipefail

usage() {
    echo "usage: bash sync_results_to_dgx.sh [--size lightweight|detailed|full] [--job-id JOB_ID]" >&2
}

SYNC_SIZE=lightweight
JOB_ID=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --size) SYNC_SIZE="$2"; shift 2 ;;
        --job-id) JOB_ID="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) usage; echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
case "$SYNC_SIZE" in
    lightweight|detailed|full) ;;
    *) usage; echo "sync size must be lightweight, detailed, or full" >&2; exit 2 ;;
esac
if [ -n "$JOB_ID" ] && ! [[ "$JOB_ID" =~ ^[0-9]+$ ]]; then
    usage
    echo "JOB_ID must be numeric" >&2
    exit 2
fi

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
SOURCE_ROOT="$SELENA_HOST:/scratch/users/$nni/codes/$PROJECT_NAME"
DGX_OUTPUT_ROOT="$PROJECT_ROOT/outputs/selena"
DGX_LOG_ROOT="$PROJECT_ROOT/logs/selena"
mkdir -p "$DGX_OUTPUT_ROOT" "$DGX_LOG_ROOT"

OUTPUT_FILTERS=()
if [ "$SYNC_SIZE" != full ]; then
    filter_text="$(python3 "$PROJECT_ROOT/src/timebench/pipeline/artifact_selection.py" filters --size "$SYNC_SIZE")"
    mapfile -t OUTPUT_FILTERS <<< "$filter_text"
fi

SIZE_OPTIONS=()
if [ "$SYNC_SIZE" = lightweight ]; then
    SIZE_OPTIONS=(--max-size="${PUBLISH_MAX_FILE_BYTES:-100000000}" --exclude="*.pt" --exclude="*.npy" --exclude="*.cbm")
fi

echo "Pulling $PROJECT_NAME Selena outputs to DGX ($SYNC_SIZE)..."
rsync -rlptz "${SIZE_OPTIONS[@]}" --partial --prune-empty-dirs --info=progress2 \
    "${OUTPUT_FILTERS[@]}" \
    "$SOURCE_ROOT/outputs/" \
    "$DGX_OUTPUT_ROOT/"

if [ -n "$JOB_ID" ]; then
    echo "Pulling Selena logs for job $JOB_ID..."
    rsync -rlptz "${SIZE_OPTIONS[@]}" --partial --prune-empty-dirs --info=progress2 \
        '--include=*/' \
        "--include=*_${JOB_ID}_*.out" "--include=*_${JOB_ID}_*.err" \
        "--include=*_${JOB_ID}.out" "--include=*_${JOB_ID}.err" \
        "--include=*/dataset_metadata/${JOB_ID}__*" \
        '--exclude=*/workflow_status/***' '--exclude=*' \
        "$SOURCE_ROOT/logs/" \
        "$DGX_LOG_ROOT/"

    status_file_list="$(mktemp)"
    trap 'rm -f -- "$status_file_list"' EXIT
    if [[ "$SOURCE_ROOT" == *:* ]]; then
        source_host="${SOURCE_ROOT%%:*}"
        source_path="${SOURCE_ROOT#*:}"
        ssh "$source_host" \
            "find '$source_path/logs' -type f -path '*/workflow_status/*' -name '*.status' -exec grep -qxF 'slurm_job_id=$JOB_ID' {} \; -printf '%P\0'" \
            > "$status_file_list"
    else
        find "$SOURCE_ROOT/logs" \
            -type f -path '*/workflow_status/*' -name '*.status' \
            -exec grep -qxF "slurm_job_id=$JOB_ID" {} \; \
            -printf '%P\0' \
            > "$status_file_list"
    fi
    if [ -s "$status_file_list" ]; then
        rsync -rlptz "${SIZE_OPTIONS[@]}" --partial --prune-empty-dirs --from0 \
            --files-from="$status_file_list" \
            "$SOURCE_ROOT/logs/" \
            "$DGX_LOG_ROOT/"
    else
        echo "WARNING: no workflow status matched Selena job $JOB_ID" >&2
    fi
else
    echo "Pulling all Selena logs and workflow status..."
    rsync -rlptz "${SIZE_OPTIONS[@]}" --partial --info=progress2 \
        "$SOURCE_ROOT/logs/" \
        "$DGX_LOG_ROOT/"
fi

echo "SUCCESS: $SYNC_SIZE TIME outputs, requested logs, and metadata aggregates were pulled from Selena to DGX."
