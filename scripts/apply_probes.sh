#!/bin/bash
# Apply each trained probe to its own held-out evaluation cell.
#
# The probe artifacts hold fitted coefficients, not held-out scores, so this is
# the step that turns them into per-record predictions the aggregation can read
# alongside the entropy methods and P(True).
#
# Scoring runs against results/probe_inputs, the tree whose clustering supplies
# the SEP target lineage, and both probe kinds are applied in turn so their
# scores land in separately named fields on the same records.
#
# CPU only: a dot product per record.
#
# Deliberately UNTRACKED.
#
# Usage:
#   bash scripts/apply_probes.sh
#   DRY_RUN=1 bash scripts/apply_probes.sh
set -uo pipefail

cd "$(dirname "$0")/.."

IN_ROOT="${IN_ROOT:-results/probe_inputs}"
PROBE_ROOT="${PROBE_ROOT:-results/probes}"
OUT_ROOT="${OUT_ROOT:-results/probe_scored}"
REGIMES="${REGIMES:-chat default}"
DATASETS="${DATASETS:-triviaqa nqopen svamp}"
SEEDS="${SEEDS:-42 43 44}"
MODELS="${MODELS:-mistralai_Mistral-7B-Instruct-v0.3 meta-llama_Llama-3.1-8B-Instruct meta-llama_Llama-3.1-70B-Instruct}"
RUN="${RUN:-uv run --frozen python}"

done_=0; failed=0
for regime in $REGIMES; do
 for dataset in $DATASETS; do
  for model in $MODELS; do
   for seed in $SEEDS; do
    cell="$regime-$dataset-$model-s$seed"
    eval_dir=$(ls -d "$IN_ROOT"/repl-"$regime"-eval-"$dataset"-"$model"-s"$seed"-* 2>/dev/null | head -1 || true)
    if [ -z "$eval_dir" ]; then
        echo "MISSING eval cell: $cell" >&2; failed=$((failed+1)); continue
    fi
    out_dir="$OUT_ROOT/$(basename "$eval_dir")"
    mkdir -p "$out_dir"

    # Chain the two probes: accuracy first, then SEP over its output, so the
    # final file carries both score fields on every record.
    src="$eval_dir/scored.jsonl"
    stage=0
    for pair in "accuracy:accuracy_probe" "sep:semantic_entropy_probe"; do
        short="${pair%%:*}"; kind="${pair##*:}"
        probe="$PROBE_ROOT/$cell/$short.json"
        [ -f "$probe" ] || { echo "MISSING probe: $probe" >&2; failed=$((failed+1)); continue; }
        dst="$out_dir/stage$stage.jsonl"
        if [ "${DRY_RUN:-0}" = "1" ]; then
            printf '%-46s %s\n' "$cell" "$short"; stage=$((stage+1)); src="$dst"; continue
        fi
        if $RUN scripts/posthoc_sep_score_jsonl.py "$src" "$dst" \
              --probe "$probe" --expected-probe-kind "$kind" \
              > "$out_dir/$short.log" 2>&1; then
            src="$dst"; stage=$((stage+1))
        else
            echo "  FAIL $cell $short -- $out_dir/$short.log" >&2
            failed=$((failed+1)); break
        fi
    done

    if [ "${DRY_RUN:-0}" != "1" ] && [ "$stage" -eq 2 ]; then
        mv "$src" "$out_dir/scored.jsonl"
        rm -f "$out_dir"/stage*.jsonl
        done_=$((done_+1))
        printf '  ok %s\n' "$cell"
    fi
   done
  done
 done
done

echo
echo "$done_ cells scored, $failed failures"
echo "output: $OUT_ROOT/"
