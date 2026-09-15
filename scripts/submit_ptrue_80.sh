#!/bin/bash
# Fan out P(True) over the replication eval cells.
#
# P(True) is a self-evaluation, so each cell is judged by its own generator and
# its few-shot prompt is built from the *scored train* artifact for the same
# regime, dataset, model and seed. Pairing on all four matters: the prompt
# teaches the model what "True" looks like for its own outputs under this
# particular prompt format, and a train artifact from the other regime would
# carry answers of a quite different length.
#
# Input is the ladder output under the canonical backend rather than the raw
# collection, so the final artifact carries clustering and P(True) together.
#
# Usage:
#   bash scripts/submit_ptrue.sh
#   DRY_RUN=1 REGIMES=chat bash scripts/submit_ptrue.sh
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"
CODE_COMMIT="$(git rev-parse HEAD)"

[ -z "$(git status --porcelain --untracked-files=no)" ] || {
    echo "ERROR: tracked worktree is dirty." >&2; exit 2; }

BACKEND="${BACKEND:-microsoft_deberta-v2-xlarge-mnli}"
# NOT $WORK: the group is over its /home/atuin inode quota, so writes there
# fail regardless of size. Moved to /home/hpc on 2026-08-20.
LADDER_ROOT="${LADDER_ROOT:-$REPO_ROOT/results/ladder/$BACKEND}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/results/ptrue}"
REGIMES="${REGIMES:-chat default}"
N_FEW_SHOT="${N_FEW_SHOT:-20}"

declare -A REV=(
  ["mistralai/Mistral-7B-Instruct-v0.3"]="c170c708c41dac9275d15a8fff4eca08d52bab71"
  ["meta-llama/Llama-3.1-8B-Instruct"]="0e9e39f249a16976918f6564b8830bc894c89659"
  ["meta-llama/Llama-3.1-70B-Instruct"]="1605565b47bb9346c5515c34102e054115b4f98b"
)

submitted=0; skipped=0; unpaired=0
for meta in results/replication/*/job_meta.json; do
    [ -f "$meta" ] || continue
    dir="$(dirname "$meta")"; cell="$(basename "$dir")"

    read -r regime role limit model dataset seed < <(python3 - "$meta" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print(m.get("replication_regime","?"), m.get("split_role","?"),
      m.get("prompt_limit") if m.get("prompt_limit") is not None else "null",
      m.get("model_name","?"), m.get("dataset","?"), m.get("seed","?"))
PY
)
    [ "$limit" = "null" ] || { skipped=$((skipped+1)); continue; }
    [ "$role" = "eval" ]  || { skipped=$((skipped+1)); continue; }
    case " $REGIMES " in *" $regime "*) ;; *) skipped=$((skipped+1)); continue ;; esac

    src="$LADDER_ROOT/$cell/scored.jsonl"
    [ -f "$src" ] || { echo "no ladder output for $cell" >&2; unpaired=$((unpaired+1)); continue; }

    # Pair with the train artifact for the same regime/dataset/model/seed.
    # The `|| true` matters: under `set -e` an assignment inherits the exit
    # status of its command substitution, so a glob that matches nothing kills
    # the whole script rather than falling through to the unpaired branch.
    safe_model="$(printf '%s' "$model" | tr '/' '_')"
    mapfile -t train_dirs < <(ls -d results/replication/repl-${regime}-train-${dataset}-${safe_model}-s${seed}-* 2>/dev/null || true)
    # Refuse to guess. Two directories for one cell key means a rerun left its
    # predecessor in place, and the two may have been generated under different
    # prompts; silently taking the first would pair P(True) against an
    # arbitrary one. Retire the older directory first.
    if [ "${#train_dirs[@]}" -gt 1 ]; then
        echo "ERROR: ${#train_dirs[@]} train directories for ${regime}/${dataset}/${model}/s${seed}:" >&2
        printf '    %s\n' "${train_dirs[@]}" >&2
        exit 3
    fi
    train_dir="${train_dirs[0]:-}"
    [ -n "$train_dir" ] && [ -f "$train_dir/scored.jsonl" ] || {
        echo "no train artifact for $cell" >&2; unpaired=$((unpaired+1)); continue; }

    out="$OUT_ROOT/$cell"
    [ -e "$out/scored.jsonl" ] && { skipped=$((skipped+1)); continue; }

    rev="${REV[$model]:-}"
    [ -n "$rev" ] || { echo "no pinned revision for $model" >&2; unpaired=$((unpaired+1)); continue; }

    RES="--constraint=a100_80"
    [ "$model" = "meta-llama/Llama-3.1-70B-Instruct" ] && \
        RES="--constraint=a100_80 --gres=gpu:a100:4 --time=12:00:00"

    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '%-8s %-9s %-26s s%s\n' "$regime" "$dataset" "${model##*/}" "$seed"
    else
        sbatch $RES --export=ALL,\
INPUT_JSONL="$src",OUT_DIR="$out",\
FEW_SHOT_DATA="$REPO_ROOT/$train_dir/scored.jsonl",\
JUDGE_MODEL="$model",JUDGE_REVISION="$rev",\
CODE_COMMIT="$CODE_COMMIT",N_FEW_SHOT="$N_FEW_SHOT" \
            slurm/replication_ptrue_alex.sbatch
    fi
    submitted=$((submitted+1))
done

echo
echo "$submitted jobs $([ "${DRY_RUN:-0}" = "1" ] && echo 'would be submitted' || echo submitted), $skipped skipped, $unpaired unpaired"
echo "output: $OUT_ROOT/"
[ "$unpaired" -gt 0 ] && echo "WARNING: $unpaired cells had no ladder output or no train pair." || true
