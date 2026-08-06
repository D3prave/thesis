#!/bin/bash -l
#SBATCH --job-name=se-bio-recluster-v3
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/recluster_bio_v3_%j.out
#SBATCH --error=logs/recluster_bio_v3_%j.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
# ============================================================================
# SEP v3 whole-paragraph biography NLI reclustering.
#
# Wraps scripts/posthoc_recluster_jsonl.py the way phase2_bio_nli_fritz.sbatch
# does — deliberately NOT question-conditioned, because the existing bio family
# was clustered answer-only and the reseal may not change methodology — but adds
# the v3 bindings that script lacks: pinned code commit, pinned NLI model and
# tokenizer revisions, master-protocol SHA, and verification of the SOURCE
# artifact before consuming it.
#
# Runs on GPU rather than Fritz CPU purely for wall-clock; the clustering rule
# (--clustering strict, the script default) is unchanged.
# ============================================================================
set -euo pipefail
echo "=== v3 bio recluster ${SLURM_JOB_ID} on $(hostname) ==="
date --iso-8601=seconds

: "${INPUT_JSONL:?}"; : "${NLI_MODEL_NAME:?}"; : "${NLI_MODEL_REVISION:?}"
: "${NLI_TOKENIZER_REVISION:?}"; : "${CODE_COMMIT:?}"
: "${V3_MASTER:?}"; : "${V3_MASTER_SHA256:?}"
NLI_BATCH_SIZE="${NLI_BATCH_SIZE:-32}"
SE_NLI_DEVICE="${SE_NLI_DEVICE:-cuda:0}"
BIO_PROMPT_SHA_EXPECTED="a5abd78a52421385801a234ed0f2c63befe3830a4947d873697c5df8af40843a"

REPO_ROOT="${SLURM_SUBMIT_DIR}"
SE_CLUSTER="alex"
# shellcheck disable=SC1091
source "${REPO_ROOT}/slurm/_common.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SE_NLI_MODEL="${NLI_MODEL_NAME}" SE_NLI_DEVICE
export SE_NLI_REVISION="${NLI_MODEL_REVISION}"
export SE_NLI_TOKENIZER_REVISION="${NLI_TOKENIZER_REVISION}"

[ "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${CODE_COMMIT}" ] || { echo "ERROR: HEAD != CODE_COMMIT" >&2; exit 2; }
[ -z "$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=no)" ] || { echo "ERROR: dirty tracked worktree" >&2; exit 2; }
[ "$(sha256sum "${REPO_ROOT}/${V3_MASTER}" | awk '{print $1}')" = "${V3_MASTER_SHA256}" ] || { echo "ERROR: master protocol changed" >&2; exit 2; }

CACHE_DIR="${HF_HOME}/hub/models--$(printf '%s' "${NLI_MODEL_NAME}" | sed 's|/|--|')"
CACHE_REF="$(cat "${CACHE_DIR}/refs/main" 2>/dev/null || true)"
[ "${CACHE_REF}" = "${NLI_MODEL_REVISION}" ] || { echo "ERROR: cached refs/main=${CACHE_REF} != ${NLI_MODEL_REVISION}" >&2; exit 2; }

se_require_data_path "${INPUT_JSONL}"
SRC_META="$(dirname "${INPUT_JSONL}")/job_meta.json"
python - "${SRC_META}" "${INPUT_JSONL}" "${V3_MASTER_SHA256}" "${BIO_PROMPT_SHA_EXPECTED}" <<'PYSRC'
import json, statistics, sys
meta = json.load(open(sys.argv[1]))
recs = [json.loads(l) for l in open(sys.argv[2])]
assert meta["master_protocol_sha256"] == sys.argv[3], "source bound to a different master protocol"
assert meta["system_prompt_sha256"] == sys.argv[4], "source generated under a different bio system prompt"
assert meta["task"] == "bio", "source is not a bio artifact"
assert len(recs) == 500, f"expected 500 records, got {len(recs)}"
lens = [len(s) for r in recs for s in r["sampled_answers"]]
med = statistics.median(lens)
assert med >= 300, f"source median sample length {med:.0f} < 300 - collapsed long-form"
print(f"source verified: n={len(recs)} median_len={med:.0f} rev={meta['model_revision'][:8]}")
PYSRC
se_require_cached_model "${NLI_MODEL_NAME}"
se_preflight --features hf --model "${NLI_MODEL_NAME}" --data-path "${INPUT_JSONL}"

LINEAGE="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['analysis_code_commit'])" "${SRC_META}")"
SOURCE_RUN_ID="$(basename "$(dirname "${INPUT_JSONL}")")"
SAFE_NLI="$(printf '%s' "${NLI_MODEL_NAME}" | tr '/' '_' | tr -c '[:alnum:]_.-' '_')"
RUN_ID="posthoc-alex-bio-recluster-v3-${SOURCE_RUN_ID}-${SAFE_NLI}-${SLURM_JOB_ID}"
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
  "nli_model": "${NLI_MODEL_NAME}",
  "nli_model_revision": "${NLI_MODEL_REVISION}",
  "nli_tokenizer_revision": "${NLI_TOKENIZER_REVISION}",
  "nli_batch_size": ${NLI_BATCH_SIZE},
  "nli_device": "${SE_NLI_DEVICE}",
  "question_conditioned": false,
  "question_conditioned_note": "answer-only, matching the pre-existing bio family; reseal changes provenance only",
  "clustering_rule": "strict",
  "analysis_code_commit": "${LINEAGE}",
  "execution_code_commit": "${CODE_COMMIT}",
  "master_protocol": "${V3_MASTER}",
  "master_protocol_sha256": "${V3_MASTER_SHA256}",
  "system_prompt_sha256": "${BIO_PROMPT_SHA_EXPECTED}",
  "start_time": "$(date --iso-8601=seconds)"
}
META

python scripts/posthoc_recluster_jsonl.py \
    "${INPUT_JSONL}" "${OUT_DIR}/scored.jsonl" \
    --nli-model "${NLI_MODEL_NAME}" \
    --nli-device "${SE_NLI_DEVICE}" \
    --nli-batch-size "${NLI_BATCH_SIZE}" \
    --entailment-backend "${NLI_MODEL_NAME##*/}" \
    --clustering strict \
    --run-id "${RUN_ID}" \
    --nli-revision "${NLI_MODEL_REVISION}" \
    --nli-tokenizer-revision "${NLI_TOKENIZER_REVISION}" \
    --code-commit "${CODE_COMMIT}" \
    --protocol-manifest "${V3_MASTER}" \
    --protocol-manifest-sha256 "${V3_MASTER_SHA256}"

python -m semantic_entropy.metrics "${OUT_DIR}/scored.jsonl" \
    --output-dir "${OUT_DIR}/metrics" --table-dir "${OUT_DIR}/tables"
echo "=== complete: ${OUT_DIR} ==="
date --iso-8601=seconds
