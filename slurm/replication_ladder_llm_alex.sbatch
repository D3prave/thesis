#!/bin/bash -l
#SBATCH --job-name=se-ladder-llm
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:4
#SBATCH --constraint=a100_80
#SBATCH --time=12:00:00
#SBATCH --output=logs/ladder_llm_%j.out
#SBATCH --error=logs/ladder_llm_%j.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
# ============================================================================
# Entailment ladder rung judged by a generative LLM instead of a cross-encoder.
#
# Farquhar et al. use GPT-3.5 for the sentence-length setting. Paid API access
# is out of scope here, so this rung substitutes an open model of comparable
# size. It is a substitution, not a reproduction.
#
# Takes the same variable names as replication_ladder_alex.sbatch so
# scripts/submit_ladder.sh drives it unchanged via LAUNCHER=.
#
# Deliberately UNTRACKED during the frozen collection scope.
# ============================================================================
set -euo pipefail

echo "=== LLM-judge ladder ${SLURM_JOB_ID} on $(hostname) ==="
date --iso-8601=seconds

: "${INPUT_JSONL:?}"; : "${OUTPUT_JSONL:?}"
: "${NLI_MODEL:?judge model id}"; : "${NLI_REVISION:?}"; : "${CODE_COMMIT:?}"

for sha in "${NLI_REVISION}" "${CODE_COMMIT}"; do
    [[ "${sha}" =~ ^[0-9a-f]{40}$ ]] || { echo "ERROR: revisions must be 40-hex" >&2; exit 2; }
done
[ -f "${INPUT_JSONL}" ] || { echo "ERROR: missing ${INPUT_JSONL}" >&2; exit 2; }
[ -e "${OUTPUT_JSONL}" ] && { echo "ERROR: refusing to overwrite ${OUTPUT_JSONL}" >&2; exit 2; }

REPO_ROOT="${SLURM_SUBMIT_DIR}"
SE_CLUSTER="alex"
# shellcheck disable=SC1091
source "${REPO_ROOT}/slurm/_common.sh"

[ "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${CODE_COMMIT}" ] || { echo "ERROR: HEAD != CODE_COMMIT" >&2; exit 2; }
[ -z "$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=no)" ] || { echo "ERROR: dirty tracked worktree" >&2; exit 2; }

se_require_cached_model "${NLI_MODEL}"

GPU_COUNT="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || echo 1)"
echo "--- ${GPU_COUNT} GPU(s) visible ---"
case "${NLI_MODEL}" in
    *70B*|*72B*)
        [ "${GPU_COUNT}" -ge 4 ] || {
            echo "ERROR: ${NLI_MODEL} needs at least 4 cards; got ${GPU_COUNT}" >&2; exit 2; }
        ;;
esac

BACKEND_LABEL="${BACKEND_LABEL:-$(printf '%s' "${NLI_MODEL}" | tr 'A-Z/' 'a-z_')}"
RUN_ID="repl-ladder-llm-$(basename "$(dirname "${OUTPUT_JSONL}")")-${SLURM_JOB_ID}"

mkdir -p "$(dirname "${OUTPUT_JSONL}")"

python scripts/posthoc_recluster_llm_judge.py \
    "${INPUT_JSONL}" \
    "${OUTPUT_JSONL}" \
    --judge-model "${NLI_MODEL}" \
    --judge-revision "${NLI_REVISION}" \
    --entailment-backend "${BACKEND_LABEL}" \
    --cluster-source raw \
    --tensor-parallel-size "${GPU_COUNT}" \
    --run-id "${RUN_ID}"

echo "--- verifying the output ---"
python - "${OUTPUT_JSONL}" <<'CHECK'
import json, sys, statistics
sizes, backends = [], set()
with open(sys.argv[1], encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        sizes.append(len(set(record["semantic_clusters"])))
        backends.add(record.get("entailment_backend"))
if not sizes:
    raise SystemExit("no records written")
# A judge that answers the same way every time collapses every record to one
# cluster (or splits every record into ten). Either is a broken run that would
# still look well-formed.
if min(sizes) == max(sizes):
    raise SystemExit(
        f"every record produced exactly {sizes[0]} cluster(s); the judge is "
        "not discriminating"
    )
print(
    f"{len(sizes)} records, clusters per record: mean "
    f"{statistics.fmean(sizes):.2f}, min {min(sizes)}, max {max(sizes)}"
)
print("backend label:", backends)
CHECK

echo "=== done $(date --iso-8601=seconds) ==="
