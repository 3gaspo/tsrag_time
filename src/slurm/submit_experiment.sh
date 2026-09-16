#!/bin/bash
set -euo pipefail
cd "$PROJECT_ROOT"
cluster="${1:-dgx}"
case "$cluster" in
    dgx)
        export TIME_STORAGE_ROOT="${TIME_STORAGE_ROOT:-$HOME}"
        source "$PROJECT_ROOT/src/slurm/runtime_paths.sh"
        front=experiment.slurm
        ;;
    selena)
        source "$PROJECT_ROOT/src/slurm/selena_runtime.sh"
        front=experiment_selena.slurm
        ;;
    *) echo 'usage: bash submit_experiment.sh dgx|selena [Hydra overrides...]' >&2; exit 2 ;;
esac
shift || true
dependency=()
[ -z "${SBATCH_DEPENDENCY:-}" ] || dependency=(--dependency="$SBATCH_DEPENDENCY")
mkdir -p "$TIME_LOGS"
sbatch "${dependency[@]}" "$front" "$@"
