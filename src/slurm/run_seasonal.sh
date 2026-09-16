#!/bin/bash
set -euo pipefail
source "$PROJECT_ROOT/src/slurm/runtime_paths.sh"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TSRAG_DEFER_COMPLETION=1
mkdir -p "$TIME_LOGS"
trap 'uv run --no-sync python -m timebench.scripts.finalize_stage "$TIME_SEASONAL_TASKS_ROOT" "$TIME_LAUNCH_ID" --interrupt' EXIT
srun --ntasks=1 uv run --no-sync python -m timebench.scripts.seasonal_grid "$@" "experiment_mode=${EXPERIMENT_MODE:-full}"
uv run --no-sync python -m timebench.scripts.finalize_stage "$TIME_SEASONAL_TASKS_ROOT" "$TIME_LAUNCH_ID"
trap - EXIT
