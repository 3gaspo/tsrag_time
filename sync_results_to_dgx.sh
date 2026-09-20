#!/bin/bash
set -euo pipefail
size=lightweight
job_id=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --size) size="$2"; shift 2 ;;
        --job-id) job_id="$2"; shift 2 ;;
        -h|--help) echo 'usage: bash sync_results_to_dgx.sh [--size lightweight|detailed|full] [--job-id JOB_ID]'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
case "$size" in lightweight|detailed|full) ;; *) echo 'invalid size' >&2; exit 2 ;; esac
[[ -z "$job_id" || "$job_id" =~ ^[0-9]+$ ]] || { echo 'JOB_ID must be numeric' >&2; exit 2; }
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
PROJECT_NAME="$(basename "$PROJECT_ROOT")"
nni="$(sed -n '1p' "${TIME_NNI_FILE:-$HOME/codes/.secrets/nni}" | tr -d '[:space:]')"
nni="${nni,,}"
[[ "$nni" =~ ^[a-z][a-z0-9_-]*$ ]] || { echo 'invalid NNI' >&2; exit 2; }
host="${TIME_SELENA_HOST:-$nni@selena.hpc.edf.fr}"
source_root="$host:/scratch/users/$nni/codes/$PROJECT_NAME"
mkdir -p "$PROJECT_ROOT/outputs/selena" "$PROJECT_ROOT/logs/selena"
filters=()
if [ "$size" != full ]; then
    filter_text="$(python3 "$PROJECT_ROOT/src/timebench/pipeline/artifact_selection.py" filters --size "$size")"
    mapfile -t filters <<< "$filter_text"
fi
SIZE_OPTIONS=()
if [ "$size" = lightweight ]; then
    SIZE_OPTIONS=(--max-size="${PUBLISH_MAX_FILE_BYTES:-100000000}" --exclude="*.pt" --exclude="*.npy" --exclude="*.cbm")
fi
rsync -rlptz "${SIZE_OPTIONS[@]}" --partial --prune-empty-dirs "${filters[@]}" "$source_root/outputs/" "$PROJECT_ROOT/outputs/selena/"
log_filters=()
if [ -n "$job_id" ]; then
    log_filters=('--include=*/' "--include=*_${job_id}.out" "--include=*_${job_id}.err" '--exclude=*')
fi
rsync -rlptz "${SIZE_OPTIONS[@]}" --partial --prune-empty-dirs "${log_filters[@]}" "$source_root/logs/" "$PROJECT_ROOT/logs/selena/"
echo "Pulled $PROJECT_NAME Selena artifacts ($size) into its DGX checkout."
