#!/bin/bash
# Phase 4 fan-out: 18 NLI reclusters + 9 judge clusters + 9 correctness gradings.
# Dispatch ONLY from an empty queue: the completeness guard reads on-disk dirs
# and is blind to queued jobs (that caused the 144-job double submission).
set -euo pipefail
cd "$(dirname "$0")"

MASTER=sep_v3_protocol.json
MSHA=$(sha256sum "$MASTER" | awk '{print $1}')
CODE_COMMIT=221d0de26e8756631040aea66433958f2ad9230a
ROOT=results_sep_v3_bio
PROMPT_SHA=a5abd78a52421385801a234ed0f2c63befe3830a4947d873697c5df8af40843a

N=$(squeue -u "$USER" -h 2>/dev/null | wc -l)
[ "$N" -eq 0 ] || { echo "ERROR: $N jobs still queued/running - dispatch only from an empty queue" >&2; exit 2; }

echo "=== guard: all 9 generation cells complete and bound ==="
python3 - "$ROOT" "$MSHA" "$PROMPT_SHA" <<'PY'
import json, glob, os, statistics as st, sys
root, msha, psha = sys.argv[1:4]
dirs = sorted(glob.glob(f"{root}/phase1/phase1-alex-bio-v3-*/"))
assert len(dirs) == 9, f"expected 9 generation cells, found {len(dirs)}"
seen = set()
for d in dirs:
    m = json.load(open(os.path.join(d, "job_meta.json")))
    recs = [json.loads(l) for l in open(os.path.join(d, "scored.jsonl"))]
    assert len(recs) == 500, f"{d}: {len(recs)} records"
    assert m["master_protocol_sha256"] == msha, f"{d}: master mismatch"
    assert m["system_prompt_sha256"] == psha, f"{d}: prompt mismatch"
    med = st.median([len(s) for r in recs for s in r["sampled_answers"]])
    assert med >= 300, f"{d}: median {med:.0f} < 300"
    seen.add((m["model_name"], m["seed"]))
assert len(seen) == 9, f"expected 3 models x 3 seeds, got {sorted(seen)}"
print(f"OK: 9 cells, 4500 records, {len(seen)} distinct (model, seed)")
PY

echo "=== revisions from the master (fail closed if undeclared) ==="
eval "$(python3 - "$MASTER" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
b = {x["backend_id"]: x for x in d["method_registry"]["clustering_backends"]}
nli = d.get("nli") or {}
def tok(model_id, fallback):
    for k in ("tokenizer_revisions", "tokenizers", "tokenizer"):
        v = nli.get(k)
        if isinstance(v, dict) and model_id in v:
            return v[model_id]
    for entry in (nli.get("models") or []):
        if isinstance(entry, dict) and entry.get("model_id") == model_id:
            r = entry.get("tokenizer_revision")
            if r: return r
    print(f"echo 'NOTE: no tokenizer revision declared for {model_id}; using model revision' >&2")
    return fallback
for key, var in (("nli-deberta-v3-base","BASE"), ("nli-deberta-v3-large","LARGE"), ("llm-judge","JUDGE")):
    e = b[key]
    print(f'{var}_ID="{e["model_id"]}"')
    print(f'{var}_REV="{e["model_revision"]}"')
    if var != "JUDGE":
        print(f'{var}_TOKREV="{tok(e["model_id"], e["model_revision"])}"')
PY
)"
echo "  base=$BASE_ID@${BASE_REV:0:8} tok=${BASE_TOKREV:0:8}"
echo "  large=$LARGE_ID@${LARGE_REV:0:8} tok=${LARGE_TOKREV:0:8}"
echo "  judge=$JUDGE_ID@${JUDGE_REV:0:8}"

COMMON="CODE_COMMIT=$CODE_COMMIT,V3_MASTER=$MASTER,V3_MASTER_SHA256=$MSHA,SE_RESULTS_ROOT=$ROOT"
n=0
for d in $ROOT/phase1/phase1-alex-bio-v3-*/; do
  IN="${d}scored.jsonl"
  sbatch --export=ALL,INPUT_JSONL=$IN,NLI_MODEL_NAME=$BASE_ID,NLI_MODEL_REVISION=$BASE_REV,NLI_TOKENIZER_REVISION=$BASE_TOKREV,$COMMON \
      recluster_bio_v3_alex.sbatch >/dev/null && n=$((n+1))
  sbatch --export=ALL,INPUT_JSONL=$IN,NLI_MODEL_NAME=$LARGE_ID,NLI_MODEL_REVISION=$LARGE_REV,NLI_TOKENIZER_REVISION=$LARGE_TOKREV,$COMMON \
      recluster_bio_v3_alex.sbatch >/dev/null && n=$((n+1))
  sbatch --export=ALL,INPUT_JSONL=$IN,JUDGE_MODEL=$JUDGE_ID,JUDGE_REVISION=$JUDGE_REV,MODE=equivalence,$COMMON \
      judge_cluster_bio_v3_alex.sbatch >/dev/null && n=$((n+1))
  sbatch --export=ALL,INPUT_JSONL=$IN,JUDGE_MODEL=$JUDGE_ID,JUDGE_REVISION=$JUDGE_REV,$COMMON \
      grade_bio_v3_alex.sbatch >/dev/null && n=$((n+1))
done
echo "=== dispatched $n jobs (expected 36) ==="
squeue -u "$USER" -o "%.10i %.20j %.3t %.4C %R"
