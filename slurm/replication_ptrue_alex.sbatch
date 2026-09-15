#!/bin/bash -l
#SBATCH --job-name=se-repl-ptrue
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/repl_ptrue_%j.out
#SBATCH --error=logs/repl_ptrue_%j.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
# ============================================================================
# P(True) for the replication grid.
#
# P(True) is a SELF-evaluation: the judge must be the generator that produced
# the answers, so JUDGE_MODEL is the cell's own model. Scoring one model's
# answers with another model's judgement would measure something else
# entirely, which is why score_ptrue_jsonl.py refuses it without an explicit
# escape hatch.
#
# The few-shot prompt is built from the *scored train* artifact for the same
# cell: Farquhar et al. construct it by running the model on 20 training
# questions and labelling each with its own correctness, so the prompt teaches
# the model what "True" looks like for its own outputs. Their released default
# is --p_true_num_fewshot 20, which is reproduced here; the prompt is truncated
# if it would exceed the context limit, exactly as their
# construct_few_shot_prompt does.
#
# ============================================================================
set -euo pipefail

echo "=== replication P(True) ${SLURM_JOB_ID} on $(hostname) ==="
date --iso-8601=seconds

: "${INPUT_JSONL:?}"; : "${OUT_DIR:?}"
: "${FEW_SHOT_DATA:?scored train jsonl for the same cell}"
: "${JUDGE_MODEL:?}"; : "${JUDGE_REVISION:?}"; : "${CODE_COMMIT:?}"
N_FEW_SHOT="${N_FEW_SHOT:-20}"

for sha in "${JUDGE_REVISION}" "${CODE_COMMIT}"; do
    [[ "${sha}" =~ ^[0-9a-f]{40}$ ]] || { echo "ERROR: revisions must be 40-hex" >&2; exit 2; }
done
[ -f "${INPUT_JSONL}" ]  || { echo "ERROR: missing ${INPUT_JSONL}" >&2; exit 2; }
[ -f "${FEW_SHOT_DATA}" ] || { echo "ERROR: missing ${FEW_SHOT_DATA}" >&2; exit 2; }
[ -e "${OUT_DIR}/scored.jsonl" ] && { echo "ERROR: refusing to overwrite ${OUT_DIR}" >&2; exit 2; }

REPO_ROOT="${SLURM_SUBMIT_DIR}"
SE_CLUSTER="alex"
# shellcheck disable=SC1091
source "${REPO_ROOT}/slurm/_common.sh"

[ "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${CODE_COMMIT}" ] || { echo "ERROR: HEAD != CODE_COMMIT" >&2; exit 2; }
[ -z "$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=no)" ] || { echo "ERROR: dirty tracked worktree" >&2; exit 2; }

se_require_cached_model "${JUDGE_MODEL}"

# vLLM shards across GPUs only if told to. score_ptrue_jsonl.py defaults
# SE_TENSOR_PARALLEL_SIZE to 1, so a 70B judge handed four cards still tries to
# load every parameter onto one and dies during engine startup. Requesting the
# cards is not the same as using them: derive the degree from what was actually
# allocated rather than trusting the submitter to set both.
GPU_COUNT="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || echo 1)"
[ "${GPU_COUNT}" -ge 1 ] || GPU_COUNT=1
export SE_TENSOR_PARALLEL_SIZE="${SE_TENSOR_PARALLEL_SIZE:-${GPU_COUNT}}"
echo "--- ${GPU_COUNT} GPU(s) visible, tensor_parallel_size=${SE_TENSOR_PARALLEL_SIZE} ---"

case "${JUDGE_MODEL}" in
    *70B*|*72B*)
        [ "${SE_TENSOR_PARALLEL_SIZE}" -ge 4 ] || {
            echo "ERROR: ${JUDGE_MODEL} needs at least 4 cards; got ${SE_TENSOR_PARALLEL_SIZE}." >&2
            echo "       submit with --constraint=a100_80 --gres=gpu:a100:4" >&2
            exit 2
        }
        ;;
esac

mkdir -p "${OUT_DIR}"

RUN_ID="repl-ptrue-$(basename "$(dirname "${INPUT_JSONL}")")-${SLURM_JOB_ID}"

echo "--- P(True) with self-judge ${JUDGE_MODEL}, ${N_FEW_SHOT}-shot ---"
python scripts/score_ptrue_jsonl.py \
    "${INPUT_JSONL}" \
    "${OUT_DIR}/scored.jsonl" \
    --judge-model "${JUDGE_MODEL}" \
    --judge-revision "${JUDGE_REVISION}" \
    --few-shot-data "${FEW_SHOT_DATA}" \
    --n-few-shot "${N_FEW_SHOT}" \
    --run-id "${RUN_ID}"

echo "--- verifying the score landed ---"
python - "${OUT_DIR}/scored.jsonl" <<'CHECK'
import json, sys, statistics
vals, missing = [], 0
with open(sys.argv[1], encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        scores = json.loads(line).get("scores", {})
        key = next((k for k in scores if "p_true" in k or "ptrue" in k), None)
        if key is None:
            missing += 1
        else:
            vals.append(float(scores[key]))
if missing:
    raise SystemExit(f"{missing} records carry no P(True) score")
# An identity check cannot catch a run that is well-formed but wrong. If the
# judge returned the same value everywhere, the True/False logprobs were never
# found and every record fell through to the 0.5 default.
distinct = len(set(round(v, 6) for v in vals))
if distinct <= 2:
    raise SystemExit(
        f"P(True) took only {distinct} distinct value(s) across {len(vals)} "
        "records; the judge is not discriminating (check the True/False "
        "token match)"
    )
print(
    f"P(True) on {len(vals)} records, mean {statistics.fmean(vals):.3f}, "
    f"{distinct} distinct values, "
    f"range {min(vals):.3f}-{max(vals):.3f}"
)
CHECK

echo "=== done $(date --iso-8601=seconds) ==="
