#!/bin/bash
set -euo pipefail
source "$PROJECT_ROOT/src/slurm/runtime_paths.sh"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TSRAG_DEFER_COMPLETION=1
stages="${STAGES:-prepare,vanilla,extract,predict,mix,evaluate,report}"
mode="${EXPERIMENT_MODE:-full}"
case "$mode" in test|full) ;; *) echo 'EXPERIMENT_MODE must be test or full' >&2; exit 2 ;; esac
trap 'uv run --no-sync python -m timebench.scripts.finalize_stage "$TIME_OUTPUTS/tsrag" "$TIME_LAUNCH_ID" --interrupt' EXIT
echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Slurm=$SLURM_JOB_ID launch=$TIME_LAUNCH_ID stages=$stages mode=$mode"
IFS=',' read -r -a selected_stages <<< "$stages"
for stage in "${selected_stages[@]}"; do
    case "$stage" in prepare|vanilla|extract|predict|mix|evaluate|report) ;; *) echo "unknown stage: $stage" >&2; exit 2 ;; esac
    echo
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] stage=$stage"
    srun --ntasks=1 uv run --no-sync python -m timebench.scripts.run_experiment "$@" "experiment_mode=$mode" "stage=$stage"
    uv run --no-sync python -m timebench.scripts.finalize_stage "$TIME_OUTPUTS/tsrag" "$TIME_LAUNCH_ID"
done
trap - EXIT
