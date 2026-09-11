#!/bin/bash -l
#SBATCH --job-name=thesis-robustness-pairs
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --output=pair_cache_%j.out
#SBATCH --error=pair_cache_%j.err
#SBATCH --export=NONE
set -euo pipefail
unset SLURM_EXPORT_ENV
REPO_ROOT=/home/hpc/b192aa/b192aa36/thesis
source slurm/_common.sh
export PYTHONPATH="${PWD}/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VALIDATION_COMMIT
VALIDATION_COMMIT=$(cat SOURCE_VERSION.txt)
python scripts/cache_robustness_pairs.py clustering_inputs.json pair_cache.json.gz
