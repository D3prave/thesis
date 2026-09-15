#!/bin/bash -l
#SBATCH --job-name=se-relabel
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:4
#SBATCH --constraint=a100_80
#SBATCH --time=02:00:00
#SBATCH --output=logs/relabel_%j.out
#SBATCH --error=logs/relabel_%j.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
# ============================================================================
# Recompute correctness labels with the LLM judge their code supports.
#
# One judge for every cell, so no model grades its own answers.
# ============================================================================
set -euo pipefail

echo "=== relabel ${SLURM_JOB_ID} on $(hostname) ==="
date --iso-8601=seconds

: "${INPUT_JSONL:?}"; : "${OUTPUT_JSONL:?}"
: "${JUDGE_MODEL:?}"; : "${JUDGE_REVISION:?}"

[[ "${JUDGE_REVISION}" =~ ^[0-9a-f]{40}$ ]] || { echo "ERROR: revision must be 40-hex" >&2; exit 2; }
[ -f "${INPUT_JSONL}" ] || { echo "ERROR: missing ${INPUT_JSONL}" >&2; exit 2; }
[ -e "${OUTPUT_JSONL}" ] && { echo "ERROR: refusing to overwrite ${OUTPUT_JSONL}" >&2; exit 2; }

REPO_ROOT="${SLURM_SUBMIT_DIR}"
SE_CLUSTER="alex"
# shellcheck disable=SC1091
source "${REPO_ROOT}/slurm/_common.sh"

se_require_cached_model "${JUDGE_MODEL}"

GPU_COUNT="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || echo 1)"
case "${JUDGE_MODEL}" in
    *70B*|*72B*)
        [ "${GPU_COUNT}" -ge 4 ] || { echo "ERROR: needs 4 cards, got ${GPU_COUNT}" >&2; exit 2; }
        ;;
esac

mkdir -p "$(dirname "${OUTPUT_JSONL}")"

python scripts/relabel_with_llm_judge.py \
    "${INPUT_JSONL}" "${OUTPUT_JSONL}" \
    --judge-model "${JUDGE_MODEL}" \
    --judge-revision "${JUDGE_REVISION}" \
    --tensor-parallel-size "${GPU_COUNT}"

echo "=== done $(date --iso-8601=seconds) ==="
