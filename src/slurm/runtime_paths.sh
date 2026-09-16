#!/bin/bash

# Shared by direct TIME run scripts and scheduler wrappers.
runtime_project_root="${PROJECT_ROOT:-${ROOT_DIR:?ROOT_DIR or PROJECT_ROOT must be set}}"

# Project path settings apply before defaults; explicit submission overrides win.
runtime_path_variables=(TIME_STORAGE_ROOT TIME_DATA_ROOT TIME_DATASET TIME_METADATA TIME_WEIGHTS
    TIME_SEASONAL_SCOPE TIME_SEASONAL_ROOT TIME_SEASONAL_TASKS_ROOT
    OUTPUTS_ROOT LOGS_ROOT TIME_OUTPUTS TIME_LOGS
    HF_HOME HUGGINGFACE_HUB_CACHE HF_DATASETS_CACHE TRANSFORMERS_CACHE TORCH_HOME)
declare -A runtime_path_overrides=()
for runtime_path_variable in "${runtime_path_variables[@]}"; do
    if [[ -v "$runtime_path_variable" ]]; then
        runtime_path_overrides["$runtime_path_variable"]="${!runtime_path_variable}"
    fi
done
if [ -f "$runtime_project_root/.env" ]; then
    runtime_allexport=false
    [[ "$-" == *a* ]] && runtime_allexport=true
    set -a
    source "$runtime_project_root/.env"
    [ "$runtime_allexport" = true ] || set +a
fi
for runtime_path_variable in "${!runtime_path_overrides[@]}"; do
    printf -v "$runtime_path_variable" '%s' "${runtime_path_overrides[$runtime_path_variable]}"
    export "$runtime_path_variable"
done
unset runtime_path_variable runtime_path_variables runtime_path_overrides runtime_allexport

TIME_STORAGE_ROOT="${TIME_STORAGE_ROOT:-$runtime_project_root}"
TIME_DATA_ROOT="${TIME_DATA_ROOT:-$TIME_STORAGE_ROOT/datasets}"
TIME_DATASET="${TIME_DATASET:-$TIME_DATA_ROOT/hf_dataset}"
TIME_METADATA="${TIME_METADATA:-$TIME_DATA_ROOT/time_metadata}"
TIME_WEIGHTS="${TIME_WEIGHTS:-$TIME_STORAGE_ROOT/weights}"
# Artifacts belong to this project; copied .env files and inherited shells must
# never route TS-RAG runs into another TIME project's outputs or logs.
runtime_artifact_root="$runtime_project_root"
if [ -n "${SELENA_NNI:-}" ]; then
    runtime_artifact_root="/scratch/users/${SELENA_NNI,,}/codes/$(basename "$runtime_project_root")"
    export TIME_SCRATCH_ROOT="$runtime_artifact_root"
fi
OUTPUTS_ROOT="$runtime_artifact_root/outputs"
LOGS_ROOT="$runtime_artifact_root/logs"
TIME_OUTPUTS="$OUTPUTS_ROOT"
TIME_LOGS="$LOGS_ROOT"

TIME_SEASONAL_SCOPE="${TIME_SEASONAL_SCOPE:-shared}"
case "$TIME_SEASONAL_SCOPE" in
    shared)
        default_seasonal_root="$TIME_STORAGE_ROOT/codes/seasonal"
        ;;
    project)
        default_seasonal_root="$TIME_OUTPUTS"
        ;;
    *)
        echo "TIME_SEASONAL_SCOPE must be shared or project" >&2
        return 2 2>/dev/null || exit 2
        ;;
esac
TIME_SEASONAL_ROOT="${TIME_SEASONAL_ROOT:-$default_seasonal_root}"
TIME_SEASONAL_TASKS_ROOT="${TIME_SEASONAL_TASKS_ROOT:-$TIME_SEASONAL_ROOT/foundation_models/tasks}"

HF_HOME="${HF_HOME:-$TIME_WEIGHTS/huggingface}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
TORCH_HOME="${TORCH_HOME:-$TIME_WEIGHTS/torch}"

export TIME_STORAGE_ROOT TIME_DATA_ROOT TIME_DATASET TIME_METADATA TIME_WEIGHTS
export TIME_SEASONAL_SCOPE TIME_SEASONAL_ROOT TIME_SEASONAL_TASKS_ROOT
export OUTPUTS_ROOT LOGS_ROOT TIME_OUTPUTS TIME_LOGS
export HF_HOME HUGGINGFACE_HUB_CACHE HF_DATASETS_CACHE TRANSFORMERS_CACHE TORCH_HOME
echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] TS-RAG paths: project=$runtime_project_root outputs=$TIME_OUTPUTS logs=$TIME_LOGS"

mkdir -p "$TIME_DATA_ROOT" "$TIME_METADATA" "$TIME_WEIGHTS" "$TIME_OUTPUTS" "$TIME_LOGS"
