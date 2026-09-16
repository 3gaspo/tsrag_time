#!/bin/bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PROJECT_ROOT TSRAG_EXPERIMENT_FAMILY=ablation
source "$PROJECT_ROOT/src/slurm/submit_experiment.sh"
