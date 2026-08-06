#!/bin/bash -l
#SBATCH --job-name=se-bio-grade-v3
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:a100:4
#SBATCH --time=02:00:00
#SBATCH --output=logs/grade_bio_v3_%j.out
#SBATCH --error=logs/grade_bio_v3_%j.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
# ============================================================================
# SEP v3 whole-paragraph biography correctness grading (Qwen LLM judge).
#
# ONE job per GENERATION cell, not per (cell x backend): longform_correctness
# ._fixed_representative grades a clustering-INDEPENDENT representative, so the
# label is a property of the generation and joins onto every backend of the same
# cell. (The module's top-level docstring still describes the older
# modal-cluster policy; the code superseded it — DEEP_FINDINGS_REVERIFICATION
# 2026-06-15 §4.9.)
#
# scripts/score_longform_correctness.py takes no --judge-revision, so the
# revision is pinned the same way generation pins it: offline mode plus a
# fail-closed check that the cache's refs/main equals the protocol revision.
#
# RESOLVED decision 4: these automated labels stay; no manual labeling. The
# thesis states the LLM-judge proxy as a limitation.
# ============================================================================
set -euo pipefail
echo "=== v3 bio grading ${SLURM_JOB_ID} on $(hostname) ==="
date --iso-8601=seconds

: "${INPUT_JSONL:?}"; : "${JUDGE_MODEL:?}"; : "${JUDGE_REVISION:?}"
: "${CODE_COMMIT:?}"; : "${V3_MASTER:?}"; : "${V3_MASTER_SHA256:?}"
TP="${SE_TENSOR_PARALLEL_SIZE:-4}"
GPU_UTIL="${SE_GPU_MEMORY_UTILIZATION:-0.90}"
MAX_LEN="${SE_MAX_MODEL_LEN:-4096}"
DTYPE="${SE_VLLM_DTYPE:-auto}"
BIO_PROMPT_SHA_EXPECTED="a5abd78a52421385801a234ed0f2c63befe3830a4947d873697c5df8af40843a"
[[ "${JUDGE_REVISION}" =~ ^[0-9a-f]{40}$ ]] || { echo "ERROR: JUDGE_REVISION must be 40-hex" >&2; exit 2; }

REPO_ROOT="${SLURM_SUBMIT_DIR}"
SE_CLUSTER="alex"
# shellcheck disable=SC1091
source "${REPO_ROOT}/slurm/_common.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SE_TENSOR_PARALLEL_SIZE="${TP}" SE_GPU_MEMORY_UTILIZATION="${GPU_UTIL}"
export SE_MAX_MODEL_LEN="${MAX_LEN}" SE_VLLM_DTYPE="${DTYPE}"

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
from collections import Counter
meta = json.load(open(sys.argv[1])); recs = [json.loads(l) for l in open(sys.argv[2])]
assert meta["master_protocol_sha256"] == sys.argv[3], "source bound to a different master protocol"
assert meta["system_prompt_sha256"] == sys.argv[4], "source used a different bio system prompt"
assert meta["task"] == "bio" and len(recs) == 500, "not a complete bio artifact"
med = statistics.median([len(s) for r in recs for s in r["sampled_answers"]])
assert med >= 300, f"source median sample length {med:.0f} < 300 - collapsed long-form"
pol = Counter("most_likely_answer" if str(r.get("most_likely_answer") or "").strip()
              else "exact_surface_mode_first_tie" for r in recs)
print(f"source verified: n={len(recs)} median_len={med:.0f} representative_policy={dict(pol)}")
PYSRC
se_require_cached_model "${JUDGE_MODEL}"
se_preflight --features hf,vllm --model "${JUDGE_MODEL}" --data-path "${INPUT_JSONL}"

LINEAGE="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['analysis_code_commit'])" "${SRC_META}")"
SOURCE_RUN_ID="$(basename "$(dirname "${INPUT_JSONL}")")"
RUN_ID="posthoc-alex-bio-grade-v3-${SOURCE_RUN_ID}-${SLURM_JOB_ID}"
OUT_DIR="${REPO_ROOT}/${SE_RESULTS_ROOT:-results}/phase1/${RUN_ID}"
[ -e "${OUT_DIR}" ] && { echo "ERROR: refusing to reuse ${OUT_DIR}" >&2; exit 2; }
mkdir -p "${OUT_DIR}/metrics"

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
  "verified_cache_revision": "${CACHE_REF}",
  "label_policy": "clustering_independent_fixed_representative",
  "label_scope": "automated_llm_judge_proxy",
  "tensor_parallel_size": ${TP},
  "analysis_code_commit": "${LINEAGE}",
  "execution_code_commit": "${CODE_COMMIT}",
  "master_protocol": "${V3_MASTER}",
  "master_protocol_sha256": "${V3_MASTER_SHA256}",
  "system_prompt_sha256": "${BIO_PROMPT_SHA_EXPECTED}",
  "start_time": "$(date --iso-8601=seconds)"
}
META

python scripts/score_longform_correctness.py \
    "${INPUT_JSONL}" "${OUT_DIR}/scored.jsonl" \
    --judge-model "${JUDGE_MODEL}" \
    --tensor-parallel-size "${TP}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --max-model-len "${MAX_LEN}" \
    --dtype "${DTYPE}" \
    --enforce-eager \
    --dataset-filter bio

python - "${OUT_DIR}/scored.jsonl" <<'PYCHK'
import json, sys
from collections import Counter
recs = [json.loads(l) for l in open(sys.argv[1])]
missing = [r["prompt_id"] for r in recs if r.get("correctness_label_provenance") is None]
if missing:
    raise SystemExit(f"ERROR: {len(missing)} records ungraded, e.g. {missing[:5]}")
lab = Counter(bool(r["correctness_label"]) for r in recs)
print(f"graded {len(recs)}: correct={lab[True]} incorrect={lab[False]} "
      f"accuracy={lab[True]/len(recs):.3f}")
if lab[True] == 0 or lab[False] == 0:
    raise SystemExit("ERROR: degenerate labels - AUROC would be undefined")
PYCHK

python -m semantic_entropy.metrics "${OUT_DIR}/scored.jsonl" --output-dir "${OUT_DIR}/metrics"
echo "=== complete: ${OUT_DIR} ==="
date --iso-8601=seconds
