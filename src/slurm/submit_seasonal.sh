#!/bin/bash
set -euo pipefail
cd "$PROJECT_ROOT"
cluster="${1:-dgx}"
export TIME_SEASONAL_SCOPE="${2:-${TIME_SEASONAL_SCOPE:-shared}}"
case "$cluster" in
    dgx)
        export TIME_STORAGE_ROOT="${TIME_STORAGE_ROOT:-$HOME}"
        source "$PROJECT_ROOT/src/slurm/runtime_paths.sh"
        front=seasonal_naive.slurm
        ;;
    selena)
        source "$PROJECT_ROOT/src/slurm/selena_runtime.sh"
        front=seasonal_naive_selena.slurm
        ;;
    *) echo 'usage: bash scripts/submit_seasonal_naive.sh dgx|selena [shared|project] [Hydra overrides...]' >&2; exit 2 ;;
esac
if [ "$#" -ge 2 ]; then shift 2; elif [ "$#" -eq 1 ]; then shift; fi
dependency=()
[ -z "${SBATCH_DEPENDENCY:-}" ] || dependency=(--dependency="$SBATCH_DEPENDENCY")
mkdir -p "$TIME_LOGS" "$TIME_SEASONAL_LOGS_ROOT"
sbatch "${dependency[@]}" \
    --output="$TIME_SEASONAL_LOGS_ROOT/%x_%j.out" \
    --error="$TIME_SEASONAL_LOGS_ROOT/%x_%j.err" \
    --export="ALL,OUTPUTS_ROOT=$TIME_SEASONAL_ROOT,LOGS_ROOT=$TIME_SEASONAL_LOGS_ROOT" \
    "$front" "$@"
