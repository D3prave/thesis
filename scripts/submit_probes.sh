#!/bin/bash
# Train both probes for every replication cell.
#
# Two artifacts per cell, from the same feature and the same split:
#
#   accuracy   -- predicts correctness from the greedy hidden state. This is
#                 the paper's "embedding regression" baseline and belongs to
#                 the replication proper.
#   sep        -- predicts high semantic entropy, thresholded from the training
#                 records only. Kossen et al., this study's addition.
#
# Training is CPU-bound logistic regression over a few hundred vectors, so this
# runs on the login node rather than through Slurm.
#
# Deliberately UNTRACKED.
#
# Usage:
#   bash scripts/submit_probes.sh
#   DRY_RUN=1 bash scripts/submit_probes.sh
#   LABEL_SOURCES=accuracy_probe_correctness bash scripts/submit_probes.sh
set -uo pipefail

cd "$(dirname "$0")/.."

IN_ROOT="${IN_ROOT:-results/probe_inputs}"
OUT_ROOT="${OUT_ROOT:-results/probes}"
REGIMES="${REGIMES:-chat default}"
DATASETS="${DATASETS:-triviaqa nqopen svamp}"
SEEDS="${SEEDS:-42 43 44}"
LABEL_SOURCES="${LABEL_SOURCES:-accuracy_probe_correctness semantic_entropy_threshold}"
MODELS="${MODELS:-mistralai_Mistral-7B-Instruct-v0.3 meta-llama_Llama-3.1-8B-Instruct meta-llama_Llama-3.1-70B-Instruct}"
# uv rewrites uv.lock on a plain `uv run`, which leaves the tracked worktree
# dirty and makes every launcher that checks for cleanliness refuse to start.
RUN="${RUN:-uv run --frozen python}"
# The paper caps generation at 50 new tokens and the faithful arm uses raw
# completion, so the model routinely runs past a finished answer into the next
# demonstration and is cut at the cap; stop-sequence truncation recovers the
# answer. Measured rates here are ~40%, against a default ceiling of 1% that
# assumes EOS-terminated collection. The ceiling is therefore set explicitly and
# recorded in every probe artifact alongside the observed rate.
MAX_TRUNCATION_RATE="${MAX_TRUNCATION_RATE:-1.0}"

trained=0; skipped=0; failed=0
for regime in $REGIMES; do
 for dataset in $DATASETS; do
  for model in $MODELS; do
   for seed in $SEEDS; do
    train_dir=$(ls -d "$IN_ROOT"/repl-"$regime"-train-"$dataset"-"$model"-s"$seed"-* 2>/dev/null | head -1 || true)
    eval_dir=$(ls -d "$IN_ROOT"/repl-"$regime"-eval-"$dataset"-"$model"-s"$seed"-* 2>/dev/null | head -1 || true)
    if [ -z "$train_dir" ] || [ -z "$eval_dir" ]; then
        echo "MISSING pair: $regime/$dataset/$model/s$seed" >&2
        failed=$((failed+1)); continue
    fi

    for label in $LABEL_SOURCES; do
        short=$([ "$label" = "accuracy_probe_correctness" ] && echo accuracy || echo sep)
        out="$OUT_ROOT/$regime-$dataset-$model-s$seed/$short.json"
        if [ -e "$out" ]; then skipped=$((skipped+1)); continue; fi

        if [ "${DRY_RUN:-0}" = "1" ]; then
            printf '%-8s %-9s %-34s s%s %s\n' "$regime" "$dataset" "$model" "$seed" "$short"
            trained=$((trained+1)); continue
        fi

        mkdir -p "$(dirname "$out")"
        if $RUN scripts/train_sep_probe.py "$train_dir/scored.jsonl" \
                --eval-jsonl "$eval_dir/scored.jsonl" \
                --label-source "$label" \
                --max-truncation-rate "$MAX_TRUNCATION_RATE" \
                --output "$out" > "${out%.json}.log" 2>&1; then
            trained=$((trained+1))
            printf '  ok   %-8s %-9s %-34s s%s %s\n' "$regime" "$dataset" "$model" "$seed" "$short"
        else
            failed=$((failed+1))
            printf '  FAIL %-8s %-9s %-34s s%s %s -- %s\n' \
                "$regime" "$dataset" "$model" "$seed" "$short" "${out%.json}.log"
        fi
    done
   done
  done
 done
done

echo
echo "$trained trained, $skipped skipped, $failed failed"
echo "output: $OUT_ROOT/"
