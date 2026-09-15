#!/bin/bash -l
#SBATCH --job-name=se-bio-judge-v3
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:a100:4
#SBATCH --time=03:00:00
#SBATCH --output=logs/judge_cluster_bio_v3_%j.out
#SBATCH --error=logs/judge_cluster_bio_v3_%j.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
# ============================================================================
# SEP v3 whole-paragraph biography LLM-judge clustering (4th backend).
#
# Wraps scripts/posthoc_llm_judge_jsonl.py with the v3 bindings, and pins two
# things the script leaves to defaults and that would silently change the
# science:
#
#   MODE=equivalence  — the long-form rule ("do these two paragraphs convey the
#                       same set of facts"). The default 'entailment' is the
#                       short-form rule and must not be used on biographies.
#   SE_ENTAILMENT_CACHE — a v3-specific cache path. The shared cache is
#                       content-addressed on (model, premise, hypothesis, mode)
#                       but was populated by an UNPINNED judge snapshot, so
#                       reusing it would import decisions from a revision the
#                       protocol does not pin.
# ============================================================================
set -euo pipefail
echo "=== v3 bio judge clustering ${SLURM_JOB_ID} on $(hostname) ==="
date --iso-8601=seconds

: "${INPUT_JSONL:?}"; : "${JUDGE_MODEL:?}"; : "${JUDGE_REVISION:?}"
: "${CODE_COMMIT:?}"; : "${V3_MASTER:?}"; : "${V3_MASTER_SHA256:?}"
MODE="${MODE:-equivalence}"
JUDGE_BATCH_SIZE="${JUDGE_BATCH_SIZE:-1024}"
TP="${SE_TENSOR_PARALLEL_SIZE:-4}"
GPU_UTIL="${SE_GPU_MEMORY_UTILIZATION:-0.90}"
MAX_LEN="${SE_MAX_MODEL_LEN:-4096}"
DTYPE="${SE_VLLM_DTYPE:-auto}"
BIO_PROMPT_SHA_EXPECTED="a5abd78a52421385801a234ed0f2c63befe3830a4947d873697c5df8af40843a"

[ "${MODE}" = "equivalence" ] || { echo "ERROR: bio requires MODE=equivalence, got ${MODE}" >&2; exit 2; }
[[ "${JUDGE_REVISION}" =~ ^[0-9a-f]{40}$ ]] || { echo "ERROR: JUDGE_REVISION must be 40-hex" >&2; exit 2; }

REPO_ROOT="${SLURM_SUBMIT_DIR}"
SE_CLUSTER="alex"
# shellcheck disable=SC1091
source "${REPO_ROOT}/slurm/_common.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SE_TENSOR_PARALLEL_SIZE="${TP}" SE_GPU_MEMORY_UTILIZATION="${GPU_UTIL}"
export SE_MAX_MODEL_LEN="${MAX_LEN}" SE_VLLM_DTYPE="${DTYPE}"
export SE_ENTAILMENT_CACHE="${SE_ENTAILMENT_CACHE:-${WORK:-$HOME}/se_entailment_cache/qwen_judge_v3_${JUDGE_REVISION:0:8}.json}"
mkdir -p "$(dirname "${SE_ENTAILMENT_CACHE}")"

[ "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${CODE_COMMIT}" ] || { echo "ERROR: HEAD != CODE_COMMIT" >&2; exit 2; }
[ -z "$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=no)" ] || { echo "ERROR: dirty tracked worktree" >&2; exit 2; }
[ "$(sha256sum "${REPO_ROOT}/${V3_MASTER}" | awk '{print $1}')" = "${V3_MASTER_SHA256}" ] || { echo "ERROR: master protocol changed" >&2; exit 2; }

CACHE_DIR="${HF_HOME}/hub/models--$(printf '%s' "${JUDGE_MODEL}" | sed 's|/|--|')"
CACHE_REF="$(cat "${CACHE_DIR}/refs/main" 2>/dev/null || true)"
[ "${CACHE_REF}" = "${JUDGE_REVISION}" ] || { echo "ERROR: cached refs/main=${CACHE_REF} != ${JUDGE_REVISION}" >&2; exit 2; }

se_require_data_path "${INPUT_JSONL}"
SRC_META="$(dirname "${INPUT_JSONL}")/job_meta.json"
python - "${SRC_META}" "${INPUT_JSONL}" "${V3_MASTER_SHA256}" "${BIO_PROMPT_SHA_EXPECTED}" <<'PYSRC'
import json, statistics, sys
meta = json.load(open(sys.argv[1])); recs = [json.loads(l) for l in open(sys.argv[2])]
assert meta["master_protocol_sha256"] == sys.argv[3], "source bound to a different master protocol"
assert meta["system_prompt_sha256"] == sys.argv[4], "source used a different bio system prompt"
assert meta["task"] == "bio" and len(recs) == 500, "not a complete bio artifact"
med = statistics.median([len(s) for r in recs for s in r["sampled_answers"]])
assert med >= 300, f"source median sample length {med:.0f} < 300 - collapsed long-form"
print(f"source verified: n={len(recs)} median_len={med:.0f}")
PYSRC
se_require_cached_model "${JUDGE_MODEL}"
se_preflight --features hf,vllm --model "${JUDGE_MODEL}" --data-path "${INPUT_JSONL}"

LINEAGE="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['analysis_code_commit'])" "${SRC_META}")"
SOURCE_RUN_ID="$(basename "$(dirname "${INPUT_JSONL}")")"
JUDGE_SLUG="$(basename "${JUDGE_MODEL}" | tr '[:upper:]' '[:lower:]')"
ENTAILMENT_BACKEND="llm-judge-${JUDGE_SLUG}"
RUN_ID="posthoc-alex-bio-judge-v3-${SOURCE_RUN_ID}-${SLURM_JOB_ID}"
OUT_DIR="${REPO_ROOT}/${SE_RESULTS_ROOT:-results}/phase2/${RUN_ID}"
[ -e "${OUT_DIR}" ] && { echo "ERROR: refusing to reuse ${OUT_DIR}" >&2; exit 2; }
mkdir -p "${OUT_DIR}/metrics" "${OUT_DIR}/tables"

cat > "${OUT_DIR}/job_meta.json" <<META
{
  "slurm_job_id": "${SLURM_JOB_ID}",
  "hostname": "$(hostname)",
  "run_id": "${RUN_ID}",
  "task": "bio",
  "dataset": "bio",
  "posthoc_cluster": "alex",
  "source_jsonl": "${INPUT_JSONL}",
  "source_run_id": "${SOURCE_RUN_ID}",
  "judge_model": "${JUDGE_MODEL}",
  "judge_revision": "${JUDGE_REVISION}",
  "judge_batch_size": ${JUDGE_BATCH_SIZE},
  "clustering_mode": "${MODE}",
  "entailment_backend": "${ENTAILMENT_BACKEND}",
  "entailment_cache": "${SE_ENTAILMENT_CACHE}",
  "tensor_parallel_size": ${TP},
  "gpu_memory_utilization": ${GPU_UTIL},
  "max_model_len": ${MAX_LEN},
  "analysis_code_commit": "${LINEAGE}",
  "execution_code_commit": "${CODE_COMMIT}",
  "master_protocol": "${V3_MASTER}",
  "master_protocol_sha256": "${V3_MASTER_SHA256}",
  "system_prompt_sha256": "${BIO_PROMPT_SHA_EXPECTED}",
  "start_time": "$(date --iso-8601=seconds)"
}
META

python scripts/posthoc_llm_judge_jsonl.py \
    "${INPUT_JSONL}" "${OUT_DIR}/scored.jsonl" \
    --judge-model "${JUDGE_MODEL}" \
    --judge-revision "${JUDGE_REVISION}" \
    --tensor-parallel-size "${TP}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --max-model-len "${MAX_LEN}" \
    --dtype "${DTYPE}" \
    --enforce-eager \
    --judge-batch-size "${JUDGE_BATCH_SIZE}" \
    --entailment-backend "${ENTAILMENT_BACKEND}" \
    --mode "${MODE}" \
    --run-id "${RUN_ID}"

python -m semantic_entropy.metrics "${OUT_DIR}/scored.jsonl" \
    --output-dir "${OUT_DIR}/metrics" --table-dir "${OUT_DIR}/tables"
echo "=== complete: ${OUT_DIR} ==="
date --iso-8601=seconds
