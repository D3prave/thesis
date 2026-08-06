#!/usr/bin/env python3
"""Join v3 bio correctness labels onto every clustering backend, then summarize.

Labels are a property of the GENERATION, not the clustering: longform_correctness
._fixed_representative grades a clustering-independent representative, so one
grading pass per cell propagates to all four backends of that cell by prompt_id.
This is a join, never a re-grade. Fail-closed on any mismatch.
"""
from __future__ import annotations
import json, glob, os, re, statistics as st, sys
from collections import defaultdict
from semantic_entropy.metrics import summarize_records

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results_sep_v3_bio"
FIELDS = ["discrete_semantic_entropy", "surface_entropy", "naive_sample_entropy"]

def load(d):
    return [json.loads(l) for l in open(os.path.join(d, "scored.jsonl"))]

# ---- 1. graded artifacts -> {source_run_id: {prompt_id: label fields}} ----
labels = {}
for d in sorted(glob.glob(f"{ROOT}/phase1/*-bio-grade-v3-*/")):
    meta = json.load(open(os.path.join(d, "job_meta.json")))
    src = meta["source_run_id"]
    assert src not in labels, f"two grading artifacts for {src}"
    recs = load(d)
    assert len(recs) == 500, f"{d}: {len(recs)} records"
    m = {}
    for r in recs:
        prov = r.get("correctness_label_provenance")
        assert prov is not None, f"{d}: {r['prompt_id']} ungraded"
        m[r["prompt_id"]] = {
            "correctness_label": r["correctness_label"],
            "correctness_label_provenance": prov,
            "correctness_score": r.get("scores", {}).get("correctness_score"),
        }
    labels[src] = m
assert len(labels) == 9, f"expected 9 graded cells, found {len(labels)}"
print(f"loaded labels for {len(labels)} cells")

# ---- 2. targets: generation (exact) + every phase2 artifact of that cell ----
targets = []
for d in sorted(glob.glob(f"{ROOT}/phase1/phase1-alex-bio-v3-*/")):
    targets.append((os.path.basename(d.rstrip("/")), d, "exact"))
for d in sorted(glob.glob(f"{ROOT}/phase2/*/")):
    name = os.path.basename(d.rstrip("/"))
    meta = json.load(open(os.path.join(d, "job_meta.json")))
    b = ("nli-base" if "nli-deberta-v3-base" in name else
         "nli-large" if "nli-deberta-v3-large" in name else
         "llm-judge" if "-bio-judge-v3-" in name else None)
    assert b, f"unclassified artifact {name}"
    targets.append((meta["source_run_id"], d, b))
assert len(targets) == 36, f"expected 36 targets, found {len(targets)}"

# ---- 3. write scored.labeled.jsonl ----
written = 0
for src, d, backend in targets:
    assert src in labels, f"{d}: no labels for source {src}"
    out = os.path.join(d, "scored.labeled.jsonl")
    recs = load(d)
    lm = labels[src]
    assert {r["prompt_id"] for r in recs} == set(lm), f"{d}: prompt_id set differs from graded cell"
    with open(out, "w", encoding="utf-8") as fh:
        for r in recs:
            e = lm[r["prompt_id"]]
            r["correctness_label"] = e["correctness_label"]
            r["correctness_label_provenance"] = e["correctness_label_provenance"]
            if e["correctness_score"] is not None:
                r.setdefault("scores", {})["correctness_score"] = e["correctness_score"]
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    written += 1
print(f"wrote {written} scored.labeled.jsonl files")

# ---- 4. summarize ----
agg = defaultdict(lambda: defaultdict(list))
acc = defaultdict(list)
for src, d, backend in targets:
    model = re.search(r"(Mistral-7B|Llama-3\.1-8B|Llama-3\.1-70B)", src).group(1)
    recs = [json.loads(l) for l in open(os.path.join(d, "scored.labeled.jsonl"))]
    s = summarize_records(recs, FIELDS)
    for f in FIELDS:
        agg[(backend, model)][f].append(s["score_metrics"][f]["auroc"])
    acc[model].append(sum(r["correctness_label"] for r in recs) / len(recs))

print("\naccuracy (LLM-judge proxy, mean over seeds):")
for m, v in sorted(acc.items()):
    print(f"  {m:14s} {st.mean(v):.3f}")
print("\nAUROC, mean +/- sd over 3 seeds:")
print(f"{'backend':10s} {'model':14s} " + " ".join(f"{f[:18]:>20s}" for f in FIELDS))
for (b, m) in sorted(agg):
    row = "".join(f"{st.mean(agg[(b,m)][f]):>13.3f} +/-{st.stdev(agg[(b,m)][f]):.3f}" for f in FIELDS)
    print(f"{b:10s} {m:14s} {row}")
