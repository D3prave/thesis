#!/bin/bash -l
#SBATCH --job-name=se-repl-ladder-gpu
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/repl_ladder_gpu_%j.out
#SBATCH --error=logs/repl_ladder_gpu_%j.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
# ============================================================================
# Entailment ladder over the saved replication samples.
#
# Re-clusters one collected artifact under one entailment backend. The samples
# are never regenerated, so the backend is the only thing that varies across
# rungs of the ladder and any difference is attributable to it alone.
#
# Rungs, oldest to newest:
#   microsoft/deberta-v2-xlarge-mnli    the model their released code loads
#                                       (their Methods text names
#                                       "DeBERTa-Large-MNLI"; the code
#                                       disagrees and the code is what ran)
#   cross-encoder/nli-deberta-v3-large  current-generation cross-encoder
#   Qwen2.5-72B-Instruct                open stand-in for the GPT-3.5
#                                       entailment used in their
#                                       sentence-length runs
#
# Clustering settings reproduce their compute_uncertainty_measures.py:
#   * strict bidirectional entailment, assigned by the order-dependent anchor
#     scan, including its reassignment behaviour;
#   * raw generations as the clustering input, not normalized strings;
#   * question-conditioning as "{question} {answer}", a bare space with no
#     labels, which is what they do for a DeBERTa backend.
#
# QUESTION_FORMAT overrides that last point. It defaults to "paper", which is
# what the faithful rungs use. Set it to "labelled" for the probe-target rung:
# the probe split audit requires entailment_input_format == "question_answer_v1",
# which only the labelled format writes. That rung is a probe input, not part
# of the faithful comparison, and must be kept in its own output root.
#
# The surface-entropy baseline is scored over normalized strings regardless of
# --cluster-source, so this study's own baseline survives the faithful run.
# ============================================================================
set -euo pipefail

echo "=== replication ladder ${SLURM_JOB_ID} on $(hostname) ==="
date --iso-8601=seconds

: "${INPUT_JSONL:?}"; : "${OUTPUT_JSONL:?}"
: "${NLI_MODEL:?}"; : "${NLI_REVISION:?}"; : "${NLI_TOKENIZER_REVISION:?}"
: "${CODE_COMMIT:?}"; : "${SPLIT_ROLE:?}"

case "${SPLIT_ROLE}" in train|eval) ;; *) echo "ERROR: SPLIT_ROLE must be train|eval" >&2; exit 2 ;; esac
for sha in "${NLI_REVISION}" "${NLI_TOKENIZER_REVISION}" "${CODE_COMMIT}"; do
    [[ "${sha}" =~ ^[0-9a-f]{40}$ ]] || { echo "ERROR: revisions/commit must be 40-hex" >&2; exit 2; }
done
[ -f "${INPUT_JSONL}" ] || { echo "ERROR: missing ${INPUT_JSONL}" >&2; exit 2; }
[ -e "${OUTPUT_JSONL}" ] && { echo "ERROR: refusing to overwrite ${OUTPUT_JSONL}" >&2; exit 2; }

REPO_ROOT="${SLURM_SUBMIT_DIR}"
SE_CLUSTER="alex"
# GPU variant of the ladder. DeBERTa on CPU proved far too slow for
# sentence-length answers: 13 of 28 cells hit a 12-hour wall on Fritz, since
# entailment cost scales with sequence length. Do not mix devices within a
# rung -- CPU and GPU kernels differ in the last bits and can flip borderline
# entailment decisions, and the ladder depends on the backend being the only
# variable.
# GPU rung: CUDA modules are required.
SE_NLI_DEVICE="${SE_NLI_DEVICE:-cuda}"
# shellcheck disable=SC1091
source "${REPO_ROOT}/slurm/_common.sh"

[ "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${CODE_COMMIT}" ] || { echo "ERROR: HEAD != CODE_COMMIT" >&2; exit 2; }
[ -z "$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=no)" ] || { echo "ERROR: dirty tracked worktree" >&2; exit 2; }

BACKEND_LABEL="$(printf '%s' "${NLI_MODEL}" | tr '/' '_')"
mkdir -p "$(dirname "${OUTPUT_JSONL}")"

# Preflight: fail before the
# model loads if an extra is missing, the NLI weights are not cached, or the
# input artifact is absent. HF_HUB_OFFLINE is set by _common.sh, so an
# uncached model would otherwise fail deep inside transformers.
export SE_NLI_MODEL="${NLI_MODEL}"
export SE_NLI_DEVICE
se_require_data_path "${INPUT_JSONL}"
se_require_cached_model "${NLI_MODEL}"
se_preflight \
    --features hf \
    --model "${NLI_MODEL}" \
    --data-path "${INPUT_JSONL}"

echo "--- reclustering with ${NLI_MODEL} ---"

python "${REPO_ROOT}/scripts/posthoc_recluster_jsonl.py" \
    "${INPUT_JSONL}" \
    "${OUTPUT_JSONL}" \
    --nli-model "${NLI_MODEL}" \
    --nli-revision "${NLI_REVISION}" \
    --nli-tokenizer-revision "${NLI_TOKENIZER_REVISION}" \
    --nli-device "${SE_NLI_DEVICE:-cuda}" \
    --nli-batch-size "${NLI_BATCH_SIZE:-32}" \
    --entailment-backend "replication-${BACKEND_LABEL}" \
    --clustering anchor \
    --cluster-source raw \
    --question-format "${QUESTION_FORMAT:-paper}" \
    --question-conditioned \
    --source-split-role "${SPLIT_ROLE}" \
    --code-commit "${CODE_COMMIT}" \
    --run-id "repl-ladder-${BACKEND_LABEL}-${SLURM_JOB_ID}" \
    --job-meta-output "$(dirname "${OUTPUT_JSONL}")/job_meta.json"

echo "--- verifying recorded provenance ---"
python - "${OUTPUT_JSONL}" <<'CHECK'
import json, sys

expected = {
    "clustering_rule": "bidirectional",
    "clustering_closure": "anchor_scan",
    "cluster_input": "sampled_answers",
    "entailment_input_format": "question_answer_paper",
    "question_conditioned": True,
}
seen = 0
with open(sys.argv[1], encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, start=1):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        for key, want in expected.items():
            got = record.get(key)
            if got != want:
                raise SystemExit(
                    f"line {line_number}: {key} is {got!r}, expected {want!r}"
                )
        seen += 1
print(f"provenance OK on {seen} records")
CHECK

echo "=== done $(date --iso-8601=seconds) ==="
