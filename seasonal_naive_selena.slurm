#!/bin/bash
#SBATCH --job-name=sseasonal
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/users/%u/codes/tsrag_time/logs/%x_%j.out
#SBATCH --error=/scratch/users/%u/codes/tsrag_time/logs/%x_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=an
#SBATCH --qos=an_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80000
#SBATCH --exclusive
#SBATCH --wckey=P12CU:DATASCIENCE

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
export PROJECT_ROOT TIME_LAUNCH_ID="${TIME_LAUNCH_ID:-selena_seasonal_${SLURM_JOB_ID}}"
source "$PROJECT_ROOT/src/slurm/selena_runtime.sh"
source "$PROJECT_ROOT/src/slurm/run_seasonal.sh"
