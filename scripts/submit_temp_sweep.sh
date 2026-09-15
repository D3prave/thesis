#!/bin/bash
# Fan out the paper-faithful short-answer grid on Alex.
#
#   2 settings x 3 datasets x 3 generators x 3 seeds = 54 generation jobs
#
# The two settings are not symmetric, and that asymmetry is the point:
#
#   sentence-length : brief prompt "chat",    ZERO demonstrations
#   short-phrase    : brief prompt "default", FIVE demonstrations
#
# Supplementary Note 7 introduces the demonstrations while describing the
# short-phrase setting and states their purpose -- "which further encourages
# the LLM to predict with brevity". Measured here, five demonstrations produce
# about 12 characters on every generator tested, against 31.5 to 121.0
# zero-shot. Passing five demonstrations to the sentence-length setting
# silently collapses it into the short-phrase one; the answer-length gate in
# each job is the backstop.
#
# Llama-3.1-70B needs four 80GB cards. On a single 40GB card device_map spills
# to CPU and the job reaches roughly 20 of 100 prompts in four hours.
#
# Usage:
#   bash scripts/submit_replication_grid.sh            # submit everything
#   DRY_RUN=1 bash scripts/submit_replication_grid.sh  # print, submit nothing
#   REGIMES=chat SEEDS=42 bash scripts/submit_replication_grid.sh   # a subset
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"
CODE_COMMIT="$(git rev-parse HEAD)"

[ -z "$(git status --porcelain --untracked-files=no)" ] || {
    echo "ERROR: tracked worktree is dirty; commit or stash before submitting." >&2
    exit 2
}

REGIMES="${REGIMES:-chat default}"
DATASETS="${DATASETS:-triviaqa nqopen svamp}"
SEEDS="${SEEDS:-42 43 44}"
SPLIT_ROLES="${SPLIT_ROLES:-eval}"
LIMIT="${LIMIT:-}"

# model:revision:tier
MODELS="${MODELS:-\
mistralai/Mistral-7B-Instruct-v0.3:c170c708c41dac9275d15a8fff4eca08d52bab71:7b_8b \
meta-llama/Llama-3.1-8B-Instruct:0e9e39f249a16976918f6564b8830bc894c89659:7b_8b \
meta-llama/Llama-3.1-70B-Instruct:1605565b47bb9346c5515c34102e054115b4f98b:70b_plus}"

# Data now comes from scripts/prepare_replication_data.py, which reproduces
# their load_ds and their seeded question selection. The previous files were
# the first 500 rows of each split, and that prefix turned out to be
# systematically harder than the next 500 (TriviaQA: all nine cells, 0.082 to
# 0.114 lower accuracy), so it was not a valid sample.
#
# Their directory names differ from the pipeline's dataset labels.
prep_dir() {
    case "$1" in
        triviaqa) echo "$REPO_ROOT/data/replication/trivia_qa" ;;
        nqopen)   echo "$REPO_ROOT/data/replication/nq" ;;
        svamp)    echo "$REPO_ROOT/data/replication/svamp" ;;
        *) echo "unknown dataset $1" >&2; exit 2 ;;
    esac
}

data_path() {
    local dataset="$1" role="$2"
    # Their split names: the evaluated split is `validation`.
    case "$role" in
        eval)  echo "$(prep_dir "$dataset")/validation.jsonl" ;;
        train) echo "$(prep_dir "$dataset")/train.jsonl" ;;
        *) echo "unknown role $role" >&2; exit 2 ;;
    esac
}

# Their five demonstrations, drawn once from random_seed=10 and identical
# across our seeds. This is deliberate: their pipeline has a single
# configuration, and the seeds here are an addition for estimating sampling
# variance. Letting the seed also move the prompt prefix would confound the two.
# prepare_replication_data.py excludes these questions, and the twenty P(True)
# examples, from both collected splits.
fewshot_path() {
    echo "$(prep_dir "$1")/fewshot.json"
}

# Demonstrations belong to the short-phrase setting only.
num_few_shot() { [ "$1" = "default" ] && echo 5 || echo 0; }

submitted=0
for SPLIT_ROLE in $SPLIT_ROLES; do
 for REGIME in $REGIMES; do
  for DATASET in $DATASETS; do
    for SPEC in $MODELS; do
      IFS=: read -r NAME REV TIER <<< "$SPEC"
      for SEED in $SEEDS; do

        RES=""
        if [ "$TIER" = "70b_plus" ]; then
            RES="--constraint=a100_80 --gres=gpu:a100:4 --time=12:00:00"
        fi

        EXPORTS="ALL,REGIME=$REGIME,DATASET=$DATASET"
        EXPORTS="$EXPORTS,DATA_PATH=$(data_path "$DATASET" "$SPLIT_ROLE")"
        EXPORTS="$EXPORTS,FEWSHOT_DATA=$(fewshot_path "$DATASET")"
        EXPORTS="$EXPORTS,SPLIT_ROLE=$SPLIT_ROLE,MODEL_NAME=$NAME,MODEL_TIER=$TIER"
        EXPORTS="$EXPORTS,MODEL_REVISION=$REV,TOKENIZER_REVISION=$REV"
        EXPORTS="$EXPORTS,CODE_COMMIT=$CODE_COMMIT,SEED=$SEED,SWEEP_TEMPERATURE=$SWEEP_TEMPERATURE,SE_SEP_COLLECT_FEATURES=0"
        EXPORTS="$EXPORTS,NUM_FEW_SHOT=$(num_few_shot "$REGIME")"
        [ -n "$LIMIT" ] && EXPORTS="$EXPORTS,LIMIT=$LIMIT"

        if [ "${DRY_RUN:-0}" = "1" ]; then
            printf '%-6s %-8s %-9s %-26s s%s shots=%s %s\n' \
                "$SPLIT_ROLE" "$REGIME" "$DATASET" "${NAME##*/}" "$SEED" \
                "$(num_few_shot "$REGIME")" "$RES"
        else
            sbatch $RES --export="$EXPORTS" slurm/replication_collect_temp_alex.sbatch
        fi
        submitted=$((submitted + 1))
      done
    done
  done
 done
done

echo
echo "$submitted jobs $([ "${DRY_RUN:-0}" = "1" ] && echo 'would be submitted' || echo submitted)"
echo "commit: $CODE_COMMIT"
echo
echo "FREEZE: do not commit tracked files until collection finishes."
echo "Every job verifies HEAD against the commit above and will refuse to run"
echo "if it moves."
