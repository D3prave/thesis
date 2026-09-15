#!/bin/bash
# Fan out the entailment ladder over already-collected replication artifacts.
#
# Reclustering only: the saved samples are never regenerated, so the entailment
# backend is the single variable across rungs and any difference between them
# is attributable to it alone.
#
# Faithful backend by setting:
#
#   short-phrase    : DeBERTa cross-encoder  (their Note 7 states this)
#   sentence-length : GPT-3.5, unavailable   -> Qwen2.5-72B as the open
#                                               stand-in, run on Alex
#
# The other rungs are this study's addition: the same samples under an older
# and a newer cross-encoder, which answers whether the method's behaviour
# depends on entailment quality and how that has shifted as encoders improved.
#
#   microsoft/deberta-v2-xlarge-mnli    the model their released code loads
#   cross-encoder/nli-deberta-v3-large  current generation
#
# Clustering settings reproduce compute_uncertainty_measures.py: strict
# bidirectional entailment, order-dependent anchor scan including its
# reassignment behaviour, raw generations as input, question-conditioning as
# "{question} {answer}" with a bare space.
#
# Usage:
#   NLI_REV=<40hex> bash scripts/submit_ladder.sh
#   REGIMES=chat DRY_RUN=1 NLI_REV=<40hex> bash scripts/submit_ladder.sh
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"
CODE_COMMIT="$(git rev-parse HEAD)"

[ -z "$(git status --porcelain --untracked-files=no)" ] || {
    echo "ERROR: tracked worktree is dirty." >&2; exit 2; }

NLI_MODEL="${NLI_MODEL:-microsoft/deberta-v2-xlarge-mnli}"
: "${NLI_REV:?set NLI_REV to the pinned 40-hex revision of $NLI_MODEL}"
[[ "$NLI_REV" =~ ^[0-9a-f]{40}$ ]] || { echo "ERROR: NLI_REV must be 40-hex" >&2; exit 2; }

REGIMES="${REGIMES:-chat}"
SPLIT_ROLES="${SPLIT_ROLES:-eval}"
# NOT $WORK. The b192aa group is over its /home/atuin *inode* quota (897k files
# against a 750k soft limit, grace exhausted), so writes there fail with
# "Disk quota exceeded" regardless of how small the output is. /home/hpc has
# room. The existing rungs were moved here on 2026-08-20.
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/results/ladder}"
# Fritz CPU proved too slow: 13 of 28 cells hit the 12-hour wall, because
# entailment cost scales with answer length and sentence-length answers are
# several times longer than short-phrase ones. Default to the GPU launcher.
# Never mix devices within a rung: differing kernels can flip borderline
# entailment decisions, and the ladder depends on the backend being the only
# variable.
LAUNCHER="${LAUNCHER:-slurm/replication_ladder_alex.sbatch}"
RES="${RES:-}"
BACKEND_TAG="$(printf '%s' "$NLI_MODEL" | tr '/' '_')"

submitted=0; skipped=0
for meta in results/replication/*/job_meta.json; do
    [ -f "$meta" ] || continue
    dir="$(dirname "$meta")"
    src="$dir/scored.jsonl"
    [ -f "$src" ] || continue

    read -r regime role limit < <(python3 - "$meta" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print(m.get("replication_regime","?"), m.get("split_role","?"),
      m.get("prompt_limit") if m.get("prompt_limit") is not None else "null")
PY
)
    # Smoke and diagnostic runs carry a prompt limit; only full cells qualify.
    [ "$limit" = "null" ] || { skipped=$((skipped+1)); continue; }
    case " $REGIMES " in *" $regime "*) ;; *) skipped=$((skipped+1)); continue ;; esac
    case " $SPLIT_ROLES " in *" $role "*) ;; *) skipped=$((skipped+1)); continue ;; esac

    cell="$(basename "$dir")"
    out="$OUT_ROOT/$BACKEND_TAG/$cell/scored.jsonl"
    [ -e "$out" ] && { skipped=$((skipped+1)); continue; }

    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '%-8s %-6s %s\n' "$regime" "$role" "$cell"
    else
        sbatch --export=ALL,\
INPUT_JSONL="$REPO_ROOT/$src",OUTPUT_JSONL="$out",\
NLI_MODEL="$NLI_MODEL",NLI_REVISION="$NLI_REV",NLI_TOKENIZER_REVISION="$NLI_REV",\
CODE_COMMIT="$CODE_COMMIT",SPLIT_ROLE="$role" \
            $RES "$LAUNCHER"
    fi
    submitted=$((submitted+1))
done

echo
echo "backend : $NLI_MODEL @ $NLI_REV"
echo "$submitted jobs $([ "${DRY_RUN:-0}" = "1" ] && echo 'would be submitted' || echo submitted), $skipped skipped"
echo "output  : $OUT_ROOT/$BACKEND_TAG/"
