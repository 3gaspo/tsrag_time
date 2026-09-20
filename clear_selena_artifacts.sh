#!/bin/bash

set -euo pipefail

usage() {
    echo "usage: bash clear_selena_artifacts.sh dgx|selena" >&2
}

platform="${1:-}"
case "$platform" in
    dgx|selena) ;;
    *) usage; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$SCRIPT_DIR"
PROJECT_NAME="$(basename "$PROJECT_ROOT")"

if [ "$platform" = dgx ]; then
    artifact_roots=(
        "$PROJECT_ROOT/logs/selena"
        "$PROJECT_ROOT/outputs/selena"
    )
else
    runtime_path_variables=(
        TIME_STORAGE_ROOT TIME_SCRATCH_ROOT
        OUTPUTS_ROOT LOGS_ROOT TIME_OUTPUTS TIME_LOGS
    )
    declare -A runtime_path_overrides=()
    for runtime_path_variable in "${runtime_path_variables[@]}"; do
        if [[ -v "$runtime_path_variable" ]]; then
            runtime_path_overrides["$runtime_path_variable"]="${!runtime_path_variable}"
        fi
    done
    if [ -f "$PROJECT_ROOT/.env" ]; then
        runtime_allexport=false
        [[ "$-" == *a* ]] && runtime_allexport=true
        set -a
        source "$PROJECT_ROOT/.env"
        [ "$runtime_allexport" = true ] || set +a
    fi
    for runtime_path_variable in "${!runtime_path_overrides[@]}"; do
        printf -v "$runtime_path_variable" '%s' \
            "${runtime_path_overrides[$runtime_path_variable]}"
        export "$runtime_path_variable"
    done
    unset runtime_path_variable runtime_path_variables runtime_path_overrides runtime_allexport

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
    TIME_STORAGE_ROOT="${TIME_STORAGE_ROOT:-/scratch/users/$nni}"
    TIME_SCRATCH_ROOT="/scratch/users/$nni/codes/$PROJECT_NAME"
    OUTPUTS_ROOT="$TIME_SCRATCH_ROOT/outputs"
    LOGS_ROOT="$TIME_SCRATCH_ROOT/logs"
    artifact_roots=(
        "$LOGS_ROOT"
        "$OUTPUTS_ROOT"
    )
fi

for artifact_root in "${artifact_roots[@]}"; do
    if [ ! -d "$artifact_root" ] || [ -L "$artifact_root" ]; then
        echo "ERROR: expected a real artifact directory: $artifact_root" >&2
        exit 1
    fi
done

for artifact_root in "${artifact_roots[@]}"; do
    (
        cd "$artifact_root"
        shopt -s nullglob
        entries=(*)
        if [ "${#entries[@]}" -gt 0 ]; then
            rm -r -- "${entries[@]}"
        fi
    )
    echo "cleared $artifact_root"
done
