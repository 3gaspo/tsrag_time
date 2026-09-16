#!/bin/bash
set -euo pipefail
cd "$PROJECT_ROOT"
cluster="${1:-dgx}"
family="${TSRAG_EXPERIMENT_FAMILY:-experiment}"
case "$family" in experiment|ablation) ;; *) echo 'unknown TS-RAG experiment family' >&2; exit 2 ;; esac
case "$cluster" in
    dgx)
        export TIME_STORAGE_ROOT="${TIME_STORAGE_ROOT:-$HOME}"
        source "$PROJECT_ROOT/src/slurm/runtime_paths.sh"
        front="$family.slurm"
        ;;
    selena)
        source "$PROJECT_ROOT/src/slurm/selena_runtime.sh"
        front="${family}_selena.slurm"
        ;;
    *) echo 'usage: bash scripts/submit_experiment.sh dgx|selena [Hydra overrides...]' >&2; exit 2 ;;
esac
shift || true
dependency=()
[ -z "${SBATCH_DEPENDENCY:-}" ] || dependency=(--dependency="$SBATCH_DEPENDENCY")
mkdir -p "$TIME_LOGS"
sbatch "${dependency[@]}" "$front" "$@"
