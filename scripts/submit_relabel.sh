#!/bin/bash
# Relabel sentence-length cells with the LLM judge. UNTRACKED.
set -euo pipefail

ROOT="${ROOT:-results/replication}"
OUT_ROOT="${OUT_ROOT:-results/relabel}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen2.5-72B-Instruct}"
JUDGE_REVISION="${JUDGE_REVISION:-495f39366efef23836d0cfae4fbe635880d2be31}"
LAUNCHER="${LAUNCHER:-slurm/replication_relabel_alex.sbatch}"

shopt -s nullglob
n=0; skipped=0
for f in "${ROOT}"/repl-chat-*/scored.jsonl; do
    cell="$(basename "$(dirname "$f")")"
    out="${OUT_ROOT}/${cell}/scored_llm.jsonl"
    if [ -e "$out" ]; then skipped=$((skipped+1)); continue; fi
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "would submit: $cell"
    else
        sbatch --export=ALL,INPUT_JSONL="$PWD/$f",OUTPUT_JSONL="$PWD/$out",\
JUDGE_MODEL="$JUDGE_MODEL",JUDGE_REVISION="$JUDGE_REVISION" "$LAUNCHER" >/dev/null
    fi
    n=$((n+1))
done
echo "submitted=$n skipped=$skipped judge=$JUDGE_MODEL"
