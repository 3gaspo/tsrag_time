#!/bin/bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
export PROJECT_ROOT
source "$PROJECT_ROOT/src/slurm/submit_experiment.sh"
