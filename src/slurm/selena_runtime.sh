#!/bin/bash

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT must be set by the Slurm front}"

# Preserve submission-time overrides while loading project-local path defaults.
# The Selena code sync deliberately keeps .env on the cluster.
runtime_path_variables=(
    TIME_STORAGE_ROOT TIME_SCRATCH_ROOT TIME_DATA_ROOT TIME_DATASET TIME_METADATA TIME_WEIGHTS
    TIME_VANILLA_PREDICTIONS_PATH
    TIME_SEASONAL_SCOPE TIME_SEASONAL_ROOT TIME_SEASONAL_TASKS_ROOT
    HF_HOME HUGGINGFACE_HUB_CACHE HF_DATASETS_CACHE TRANSFORMERS_CACHE TORCH_HOME
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
    printf -v "$runtime_path_variable" '%s' "${runtime_path_overrides[$runtime_path_variable]}"
    export "$runtime_path_variable"
done
unset runtime_path_variable runtime_path_variables runtime_path_overrides runtime_allexport

module load python/3.12_pypsa || exit 1
export UV_PYTHON_DOWNLOADS=never
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
NNI_FILE="${TIME_NNI_FILE:-$HOME/codes/.secrets/nni}"
if [ ! -f "$NNI_FILE" ]; then
    echo "missing Selena NNI file: $NNI_FILE" >&2
    exit 1
fi

SELENA_NNI="$(sed -n '1p' "$NNI_FILE" | tr -d '[:space:]')"
selena_nni="${SELENA_NNI,,}"
if [[ ! "$selena_nni" =~ ^[a-z][a-z0-9_-]*$ ]]; then
    echo "the Selena NNI file must contain one valid account name" >&2
    exit 1
fi

PROJECT_NAME="$(basename "$PROJECT_ROOT")"
TIME_STORAGE_ROOT="${TIME_STORAGE_ROOT:-/scratch/users/$selena_nni}"
TIME_SCRATCH_ROOT="/scratch/users/$selena_nni/codes/$PROJECT_NAME"
TIME_DATA_ROOT="${TIME_DATA_ROOT:-$TIME_STORAGE_ROOT/datasets}"
TIME_DATASET="${TIME_DATASET:-$TIME_DATA_ROOT/hf_dataset}"
TIME_METADATA="${TIME_METADATA:-$TIME_DATA_ROOT/time_metadata}"
TIME_WEIGHTS="${TIME_WEIGHTS:-$TIME_STORAGE_ROOT/weights}"
TIME_VANILLA_PREDICTIONS_PATH="${TIME_VANILLA_PREDICTIONS_PATH:-/scratch/users/$selena_nni/codes/evaluating_tsfms/outputs/foundation_models/inference}"
OUTPUTS_ROOT="$TIME_SCRATCH_ROOT/outputs"
LOGS_ROOT="$TIME_SCRATCH_ROOT/logs"
TIME_OUTPUTS="$OUTPUTS_ROOT"
TIME_LOGS="$LOGS_ROOT"

export SELENA_NNI TIME_STORAGE_ROOT TIME_SCRATCH_ROOT
export TIME_DATA_ROOT TIME_DATASET TIME_METADATA TIME_WEIGHTS
export TIME_VANILLA_PREDICTIONS_PATH
export OUTPUTS_ROOT LOGS_ROOT TIME_OUTPUTS TIME_LOGS
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$TIME_OUTPUTS" "$TIME_LOGS"
source "$PROJECT_ROOT/src/slurm/runtime_paths.sh"
