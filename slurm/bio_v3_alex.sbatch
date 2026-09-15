#!/bin/bash -l
#SBATCH --job-name=se-bio-v3
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/bio_v3_%j.out
#SBATCH --error=logs/bio_v3_%j.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
# ============================================================================
# SEP v3 whole-paragraph biography generation.
#
# Same pipeline invocation as slurm/phase1_bio_*_alex.sbatch, with the
# provenance the v3 family requires and the older bio scripts omit:
#   * pinned code commit + clean tracked worktree;
#   * the model's cached refs/main must EQUAL the protocol revision, checked
#     fail-closed before any generation (the vLLM adapter does not accept a
#     revision, so this verification is what makes the weights attributable);
#   * master-protocol SHA, bio source hash, and top_k recorded in job_meta.
#
# Probability-weighted estimators are deliberately NOT produced here
# (SE_RETURN_LOGPROBS=0): the vLLM path exposes tempered/truncated logprobs,
# while the protocol pins raw untempered log-softmax. Bio therefore reports the
# count-based estimators only.
# ============================================================================
set -euo pipefail
echo "=== v3 bio generation ${SLURM_JOB_ID} on $(hostname) ==="
date --iso-8601=seconds

: "${MODEL_NAME:?}"; : "${MODEL_REVISION:?}"; : "${MODEL_TIER:?}"; : "${SEED:?}"
: "${DATA_PATH:?}"; : "${CODE_COMMIT:?}"
: "${V3_MASTER:?}"; : "${V3_MASTER_SHA256:?}"
NUM_SAMPLES=10; TEMPERATURE=1.0; TOP_P=0.9; TOP_K=50; MAX_NEW_TOKENS=256

REPO_ROOT="${SLURM_SUBMIT_DIR}"
SE_CLUSTER="alex"
# shellcheck disable=SC1091
source "${REPO_ROOT}/slurm/_common.sh"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SE_RETURN_LOGPROBS=0
export SE_MODEL_NAME="${MODEL_NAME}"
export SE_TEMPERATURE="${TEMPERATURE}"
export SE_TOP_P="${TOP_P}"
export SE_TOP_K="${TOP_K}"
export SE_MAX_NEW_TOKENS="${MAX_NEW_TOKENS}"
export SE_SEED="${SEED}"

# 70B requires vLLM tensor parallelism. Deliberately left UNSET for the 7B/8B
# tier so a re-run reproduces the cells already collected at adapter defaults.
TP_SIZE=""; GPU_UTIL=""; MAX_LEN=""
if [ "${MODEL_TIER}" = "70b_plus" ]; then
    TP_SIZE="${SE_TENSOR_PARALLEL_SIZE:-4}"
    GPU_UTIL="${SE_GPU_MEMORY_UTILIZATION:-0.95}"
    MAX_LEN="${SE_MAX_MODEL_LEN:-4096}"
    export SE_TENSOR_PARALLEL_SIZE="${TP_SIZE}"
    export SE_GPU_MEMORY_UTILIZATION="${GPU_UTIL}"
    export SE_MAX_MODEL_LEN="${MAX_LEN}"
    export SE_VLLM_DTYPE="${SE_VLLM_DTYPE:-auto}"
fi

# Long-form system prompt. WITHOUT THIS the adapter keeps its terse QA default
# and every "biography" comes back as one short phrase (adapters.BIO_SYSTEM_PROMPT
# documents the failure; it voided the 2026-08-01 first attempt). Read from the
# module at the pinned commit and hash-checked, so the seal records the exact
# prompt the samples were generated under and a future edit fails closed.
BIO_PROMPT_SHA_EXPECTED="a5abd78a52421385801a234ed0f2c63befe3830a4947d873697c5df8af40843a"
SE_SYSTEM_PROMPT="$(python -c 'from semantic_entropy.adapters import BIO_SYSTEM_PROMPT; print(BIO_SYSTEM_PROMPT, end="")')"
BIO_PROMPT_SHA="$(printf '%s' "${SE_SYSTEM_PROMPT}" | sha256sum | awk '{print $1}')"
[ "${BIO_PROMPT_SHA}" = "${BIO_PROMPT_SHA_EXPECTED}" ] || { echo "ERROR: BIO_SYSTEM_PROMPT changed: ${BIO_PROMPT_SHA}" >&2; exit 2; }
BIO_PROMPT_JSON="$(python -c 'import json; from semantic_entropy.adapters import BIO_SYSTEM_PROMPT; print(json.dumps(BIO_SYSTEM_PROMPT))')"
export SE_SYSTEM_PROMPT
echo "bio system prompt verified: ${BIO_PROMPT_SHA}"

[ "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${CODE_COMMIT}" ] || { echo "ERROR: HEAD != CODE_COMMIT" >&2; exit 2; }
[ -z "$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=no)" ] || { echo "ERROR: dirty tracked worktree" >&2; exit 2; }
[ "$(sha256sum "${REPO_ROOT}/${V3_MASTER}" | awk '{print $1}')" = "${V3_MASTER_SHA256}" ] || { echo "ERROR: master protocol changed" >&2; exit 2; }

# The generation path cannot take a revision argument, so verify the cache
# resolves the model to exactly the protocol revision before loading it.
CACHE_DIR="${HF_HOME}/hub/models--$(printf '%s' "${MODEL_NAME}" | sed 's|/|--|')"
CACHE_REF="$(cat "${CACHE_DIR}/refs/main" 2>/dev/null || true)"
if [ "${CACHE_REF}" != "${MODEL_REVISION}" ]; then
    echo "ERROR: cached refs/main=${CACHE_REF} != protocol revision ${MODEL_REVISION}" >&2
    exit 2
fi
echo "verified cached revision: ${CACHE_REF}"

se_require_cached_model "${MODEL_NAME}"
se_require_data_path "${DATA_PATH}"
DATA_SHA256="$(sha256sum "${DATA_PATH}" | awk '{print $1}')"

SAFE_MODEL="$(printf '%s' "${MODEL_NAME}" | tr '/' '_' | tr -c '[:alnum:]_.-' '_')"
RUN_ID="phase1-alex-bio-v3-${SAFE_MODEL}-s${SEED}-${SLURM_JOB_ID}"
OUT_DIR="${REPO_ROOT}/${SE_RESULTS_ROOT:-results}/phase1/${RUN_ID}"
[ -e "${OUT_DIR}" ] && { echo "ERROR: refusing to reuse ${OUT_DIR}" >&2; exit 2; }
mkdir -p "${OUT_DIR}"

cat > "${OUT_DIR}/job_meta.json" <<META
{
  "slurm_job_id": "${SLURM_JOB_ID}",
  "hostname": "$(hostname)",
  "task": "bio",
  "dataset": "bio",
  "split_role": "eval",
  "model_name": "${MODEL_NAME}",
  "model_revision": "${MODEL_REVISION}",
  "tokenizer_revision": "${MODEL_REVISION}",
  "verified_cache_revision": "${CACHE_REF}",
  "model_tier": "${MODEL_TIER}",
  "model_module": "semantic_entropy.adapters:make_vllm_phase1_model",
  "analysis_code_commit": "${CODE_COMMIT}",
  "execution_code_commit": "${CODE_COMMIT}",
  "master_protocol": "${V3_MASTER}",
  "master_protocol_sha256": "${V3_MASTER_SHA256}",
  "num_samples": ${NUM_SAMPLES},
  "temperature": ${TEMPERATURE},
  "top_p": ${TOP_P},
  "top_k": ${TOP_K},
  "max_new_tokens": ${MAX_NEW_TOKENS},
  "seed": ${SEED},
  "tensor_parallel_size": "${TP_SIZE}",
  "gpu_memory_utilization": "${GPU_UTIL}",
  "max_model_len": "${MAX_LEN}",
  "system_prompt": ${BIO_PROMPT_JSON},
  "system_prompt_sha256": "${BIO_PROMPT_SHA}",
  "system_prompt_source": "semantic_entropy.adapters.BIO_SYSTEM_PROMPT@${CODE_COMMIT}",
  "return_logprobs": false,
  "logprob_omission_reason": "vllm path exposes tempered/truncated logprobs; protocol pins raw untempered log-softmax",
  "data_path": "${DATA_PATH}",
  "data_artifact_sha256": "${DATA_SHA256}",
  "hf_home": "${HF_HOME}",
  "limit": null,
  "start_time": "$(date --iso-8601=seconds)"
}
META

python -m semantic_entropy.pipeline_cli \
    --dataset bio \
    --task bio \
    --run-id "${RUN_ID}" \
    --phase phase1 \
    --cluster alex \
    --model "${MODEL_NAME}" \
    --model-tier "${MODEL_TIER}" \
    --data-path "${DATA_PATH}" \
    --num-samples "${NUM_SAMPLES}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --top-k "${TOP_K}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --seed "${SEED}" \
    --model-module semantic_entropy.adapters:make_vllm_phase1_model \
    --entailment-backend exact-match \
    "${OUT_DIR}/scored.jsonl"

wc -l "${OUT_DIR}/scored.jsonl"

# Fail-closed long-form check: a collapsed short-answer run must not be sealed.
python - "${OUT_DIR}/scored.jsonl" <<'PYCHK'
import json, statistics, sys
recs = [json.loads(l) for l in open(sys.argv[1])]
lens = [len(s) for r in recs for s in r["sampled_answers"]]
med = statistics.median(lens)
print(f"sample length median={med:.0f} min={min(lens)} max={max(lens)} n={len(lens)}")
if med < 300:
    raise SystemExit(f"ERROR: median sample length {med:.0f} < 300 chars - long-form generation collapsed")
PYCHK
python -m semantic_entropy.metrics "${OUT_DIR}/scored.jsonl" --output-dir "${OUT_DIR}/metrics"
echo "=== complete: ${OUT_DIR} ==="
date --iso-8601=seconds
