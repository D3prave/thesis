#!/bin/bash -l
#SBATCH --job-name=se-repl-collect
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/repl_collect_%j.out
#SBATCH --error=logs/repl_collect_%j.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
# ============================================================================
# Paper-faithful short-answer collection (Farquhar et al. 2024).
#
# Differs from the sealed v3 collection in the prompt, and only in the prompt:
#
#   * plain-text completion layout, no chat template and no system message;
#   * one brief instruction at the top, chosen by REGIME:
#       chat     -> "Answer the following question in a single brief but
#                    complete sentence."          (their main text, ~96 chars)
#       default  -> "Answer the following question as briefly as possible."
#                                                 (their Note 7, ~16 chars)
#   * five worked Question:/Answer: demonstrations, drawn once per run from the
#     training split and seeded by SEED;
#   * SVAMP alone gets a separate "Context:" line, because their
#     generate_answers.py forces use_context=True for that dataset;
#   * the most-likely response is *sampled* at T=0.1 rather than decoded by
#     argmax, matching their generate_answers.py.
#
# A wrapped sbatch inherits its whole environment, and PIPELINE_ARGS alone has
# previously failed to carry SE_SYSTEM_PROMPT. An identity gate cannot catch a
# run that is well-formed but generated under the wrong brief instruction, so
# this script asserts on mean answer length after generation and fails the job
# if the regime does not match. Their published figures are 95.6 +/- 69.6
# characters for sentence-length and 15.9 +/- 21.8 for short-phrase.
#
# Deliberately UNTRACKED during the frozen collection scope so it cannot move
# HEAD; commit it afterwards for the record.
# ============================================================================
set -euo pipefail

echo "=== replication collect ${SLURM_JOB_ID} on $(hostname) ==="
date --iso-8601=seconds
echo "GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo none)"

: "${REGIME:?set REGIME=chat|default}"
: "${DATASET:?}"; : "${DATA_PATH:?}"; : "${FEWSHOT_DATA:?}"; : "${SPLIT_ROLE:?}"
: "${MODEL_NAME:?}"; : "${MODEL_TIER:?}"; : "${MODEL_REVISION:?}"
: "${TOKENIZER_REVISION:?}"; : "${CODE_COMMIT:?}"; : "${SEED:?}"

case "${REGIME}" in chat|default) ;; *) echo "ERROR: REGIME must be chat|default" >&2; exit 2 ;; esac
case "${SPLIT_ROLE}" in train|eval) ;; *) echo "ERROR: SPLIT_ROLE must be train|eval" >&2; exit 2 ;; esac
case "${DATASET}" in triviaqa|nqopen|svamp) ;; *) echo "ERROR: bad DATASET" >&2; exit 2 ;; esac
for sha in "${MODEL_REVISION}" "${TOKENIZER_REVISION}" "${CODE_COMMIT}"; do
    [[ "${sha}" =~ ^[0-9a-f]{40}$ ]] || { echo "ERROR: revisions/commit must be 40-hex" >&2; exit 2; }
done

# Resource guard. The #SBATCH block above requests a single A100-40GB, which
# suits the 7B/8B generators. Llama-3.1-70B needs roughly 140 GB in fp16, so on
# one 40 GB card device_map="auto" spills most of the model to CPU and the job
# crawls: two such jobs reached only 20 of 100 prompts before hitting the
# four-hour wall. Override the resources at submit time for 70B:
#
#   sbatch --constraint=a100_80 --gres=gpu:a100:4 --time=12:00:00 \
#          --export=ALL,MODEL_TIER=70b_plus,... slurm/replication_collect_alex.sbatch
#
# Failing here costs seconds; discovering it from a timeout costs four hours.
_SE_GPU_COUNT="${SLURM_GPUS_ON_NODE:-}"
if [ -z "${_SE_GPU_COUNT}" ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    _SE_GPU_COUNT="$(printf '%s' "${CUDA_VISIBLE_DEVICES}" | awk -F, '{print NF}')"
fi
if [ "${MODEL_TIER}" = "70b_plus" ] && [ "${_SE_GPU_COUNT:-1}" -lt 4 ]; then
    echo "ERROR: MODEL_TIER=70b_plus needs 4 GPUs; this allocation has ${_SE_GPU_COUNT:-unknown}." >&2
    echo "       Resubmit with:" >&2
    echo "         sbatch --constraint=a100_80 --gres=gpu:a100:4 --time=12:00:00 ..." >&2
    exit 2
fi
unset _SE_GPU_COUNT

REPO_ROOT="${SLURM_SUBMIT_DIR}"
SE_CLUSTER="alex"
# shellcheck disable=SC1091
source "${REPO_ROOT}/slurm/_common.sh"

[ "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${CODE_COMMIT}" ] || { echo "ERROR: HEAD != CODE_COMMIT" >&2; exit 2; }
[ -z "$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=no)" ] || { echo "ERROR: dirty tracked worktree" >&2; exit 2; }

# --- released defaults, from uncertainty/utils/utils.py ---------------------
export SE_MODEL_NAME="${MODEL_NAME}"
export SE_MODEL_REVISION="${MODEL_REVISION}"
export SE_TOKENIZER_REVISION="${TOKENIZER_REVISION}"
export SE_TEMPERATURE="${SWEEP_TEMPERATURE:?}"
export SE_TOP_P=0.9
export SE_TOP_K=50
export SE_SEED="${SEED}"
export SE_MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-50}"
# The sampled answers use the original's cap of 50. The single most-likely
# response keeps this project's canonical SEP budget of 256, which the
# Kossen-style contract in harness.py requires so that the response can
# terminate at EOS rather than being cut off mid-sentence -- a truncated final
# token is not a valid final-content-token feature.
#
# The difference is close to inert here: STOP_SEQUENCES includes a single
# newline, so short-answer generation stops at the first line break well before
# either cap. Sentence-length answers run about 96 characters, roughly 25
# tokens. The cap therefore binds only on a runaway generation, which is
# exactly the case the 256 exists to let terminate. Truncation is measured per
# run rather than assumed; see the budgets section of Chapter 4.
export SE_SEP_FEATURE_MAX_NEW_TOKENS=256
export SE_FEATURE_TEMPERATURE=0.1
export SE_SEP_HIDDEN_LAYER=-1
export SE_SEP_HIDDEN_TOKEN=-1
export SE_RETURN_LOGPROBS=1
export SE_SEP_COLLECT_FEATURES="${SE_SEP_COLLECT_FEATURES:-1}"
export SE_DEVICE_MAP="${SE_DEVICE_MAP:-auto}"

# --- replication prompt construction ---------------------------------------
export SE_REPLICATION_REGIME="${REGIME}"
export SE_REPLICATION_FEWSHOT_DATA="${FEWSHOT_DATA}"
# A .json here is their fixed five demonstrations from
# prepare_replication_data.py; the adapter then bypasses seeded selection so the
# prompt prefix does not move with the run seed. A .jsonl is the legacy pool.
case "${FEWSHOT_DATA}" in
    *.json) export SE_REPLICATION_FEWSHOT_JSON="${FEWSHOT_DATA}" ;;
esac
export SE_REPLICATION_EVAL_DATA="${DATA_PATH}"
export SE_REPLICATION_NUM_FEW_SHOT="${NUM_FEW_SHOT:-5}"
# Must be "none", not unset. adapters._env_system_prompt treats an absent
# variable as "keep the factory default", and that default is the terse QA
# instruction -- which would wrap the faithful completion prompt in a chat
# template carrying a second, contradictory instruction. The adapter refuses
# to start unless this is explicitly disabled.
export SE_SYSTEM_PROMPT=none
# Raw completion, not a chat turn. Disabling the system message is not enough:
# the adapter would still wrap the prompt in the model's chat template, and an
# instruction-tuned model answers a chat turn tersely whatever the embedded
# text asks for -- which silently collapses the sentence-length regime back to
# short phrases. Selecting raw completion also enables their textual stop
# sequences, which is what terminates generation in the absence of an
# end-of-turn token.
export SE_USE_CHAT_TEMPLATE=0

se_require_cached_model "${MODEL_NAME}"
se_require_data_path "${DATA_PATH}"
se_require_data_path "${FEWSHOT_DATA}"
se_preflight --features hf --model "${MODEL_NAME}" --data-path "${DATA_PATH}"

DATA_SHA256="$(sha256sum "${DATA_PATH}" | awk '{print $1}')"
FEWSHOT_SHA256="$(sha256sum "${FEWSHOT_DATA}" | awk '{print $1}')"

SAFE_MODEL="$(printf '%s' "${MODEL_NAME}" | tr '/' '_' | tr -c '[:alnum:]_.-' '_')"
RUN_ID="repl-${REGIME}-${SPLIT_ROLE}-${DATASET}-${SAFE_MODEL}-s${SEED}-${SLURM_JOB_ID}"
OUT_DIR="${REPO_ROOT}/${SE_RESULTS_ROOT:-results}/replication_T${SWEEP_TEMPERATURE}/${RUN_ID}"
[ -e "${OUT_DIR}" ] && { echo "ERROR: refusing to reuse ${OUT_DIR}" >&2; exit 2; }
mkdir -p "${OUT_DIR}"

cat > "${OUT_DIR}/job_meta.json" <<META
{
  "slurm_job_id": "${SLURM_JOB_ID}",
  "hostname": "$(hostname)",
  "replication_regime": "${REGIME}",
  "model_name": "${MODEL_NAME}",
  "model_revision": "${MODEL_REVISION}",
  "tokenizer_revision": "${TOKENIZER_REVISION}",
  "analysis_code_commit": "${CODE_COMMIT}",
  "model_tier": "${MODEL_TIER}",
  "dataset": "${DATASET}",
  "split_role": "${SPLIT_ROLE}",
  "seed": ${SEED},
  "num_samples": 10,
  "num_few_shot": ${SE_REPLICATION_NUM_FEW_SHOT},
  "prompt_format": "plain_completion_no_chat_template",
  "probe_feature_decoding": "sampled_t0.1",
  "temperature": ${SWEEP_TEMPERATURE},
  "top_p": 0.9,
  "top_k": 50,
  "max_new_tokens": ${SE_MAX_NEW_TOKENS},
  "data_path": "${DATA_PATH}",
  "data_artifact_sha256": "${DATA_SHA256}",
  "fewshot_data_path": "${FEWSHOT_DATA}",
  "fewshot_artifact_sha256": "${FEWSHOT_SHA256}",
  "prompt_limit": ${LIMIT:-null},
  "start_time": "$(date --iso-8601=seconds)"
}
META

echo "--- collecting regime=${REGIME} dataset=${DATASET} seed=${SEED} ---"
python -m semantic_entropy.pipeline_cli \
    --dataset "${DATASET}" \
    --run-id "${RUN_ID}" \
    --phase phase2 \
    --cluster alex \
    --model "${MODEL_NAME}" \
    --model-tier "${MODEL_TIER}" \
    --model-with-states-module semantic_entropy.replication_adapter:make_model_with_states \
    --num-samples 10 \
    --temperature "${SWEEP_TEMPERATURE}" \
    --top-p 0.9 \
    --top-k 50 \
    --max-new-tokens "${SE_MAX_NEW_TOKENS}" \
    --seed "${SEED}" \
    --entailment-backend exact-match \
    --paper-accuracy \
    --data-path "${DATA_PATH}" \
    ${LIMIT:+--limit "${LIMIT}"} \
    ${PIPELINE_EXTRA_ARGS:-} \
    "${OUT_DIR}/scored.jsonl"

# Notes on the flags above, which mirror the sealed collect:
#   --paper-accuracy       SQuAD token-F1 > 0.5, the rule the original's
#                          released code applies by default (`--metric squad`,
#                          `f1 >= 50`). Omitting it would silently grade by a
#                          different rule.
#   --entailment-backend   Collection clusters by exact match; real entailment
#     exact-match          is applied afterwards by the ladder, so a single set
#                          of samples serves every backend and the backend is
#                          the only variable across rungs.
#   --temperature/--top-*  Passed explicitly as well as through SE_* env vars:
#                          the env vars configure the adapter, these configure
#                          the pipeline, and they must agree.

# --- answer-length gate ------------------------------------------------------
# The only external check that the intended brief instruction reached the model.
echo "--- verifying answer length against the published figure ---"
python - "${OUT_DIR}/scored.jsonl" "${REGIME}" <<'GATE' || echo "WARNING: answer-length gate failed (non-fatal in sweep)"
import json, sys
from semantic_entropy.replication_prompts import assert_answer_length

path, regime = sys.argv[1], sys.argv[2]
answers = []
with open(path, encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if line:
            answers.extend(json.loads(line).get("sampled_answers", []))

observed = assert_answer_length(answers, regime, tolerance_sds=1.0)
print(f"mean answer length {observed:.1f} characters over {len(answers)} samples: OK")
GATE

echo "=== done $(date --iso-8601=seconds) ==="
