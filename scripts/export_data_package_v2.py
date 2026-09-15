#!/usr/bin/env python3
"""Export the measurement files in data/ from the result trees on the cluster.

Fixes over v1:
  * probe rows under squad_token_f1 read correctness_label_squad, not the
    judge label that probe_inputs swapped in (v1 emitted 54 mislabelled rows)
  * per-record answer length exported as a covariate
  * TriviaQA deduplication applied to the auxiliary files too
Adds: the four-corner factorial, the Qwen backend on the temperature sweep,
and the graders run over the short-phrase and factorial conditions.
"""
from __future__ import annotations
import csv, glob, json, re, shutil, statistics as st
from collections import defaultdict
from pathlib import Path
import numpy as np

OUT = Path("data"); OUT.mkdir(parents=True, exist_ok=True)
EXCL = set(json.loads(Path("results/dedup_triviaqa.json").read_text())["excluded_eval_prompts"])
CELL = re.compile(r"repl-(chat|default)-(train|eval)-(\w+)-(.+)-s(4[234])-\d+$")
ENTROPY = ["discrete_semantic_entropy","semantic_entropy_full","surface_entropy",
           "naive_entropy","naive_sample_entropy"]

def key(n):
    m = CELL.match(n); return m.groups() if m else None

def load_grader(root):
    out = {}
    for f in glob.glob(f"{root}/*/scored_llm.jsonl"):
        k = key(Path(f).parent.name)
        if k: out[k] = {json.loads(l)["prompt_id"]: json.loads(l)["correctness_label"]
                        for l in open(f)}
    return out

# condition -> (instruction, shots, ladder_root, {grader: relabel_root})
CONDITIONS = {
 "chat_0shot":    ("chat", 0, "results/ladder",
     {"llm_qwen2.5-72b":"results/relabel", "llm_llama-3.1-70b":"results/relabel_llama70b"}),
 "chat_5shot":    ("chat", 5, "results/ladder_factorial_chat5",
     {"llm_qwen2.5-72b":"results/relabel_factorial_chat5_qwen",
      "llm_llama-3.1-70b":"results/relabel_factorial_chat5_llama70b"}),
 "default_0shot": ("default", 0, "results/ladder_factorial_default0",
     {"llm_qwen2.5-72b":"results/relabel_factorial_default0_qwen",
      "llm_llama-3.1-70b":"results/relabel_factorial_default0_llama70b"}),
 "default_5shot": ("default", 5, "results/ladder",
     {"llm_qwen2.5-72b":"results/relabel_default_qwen",
      "llm_llama-3.1-70b":"results/relabel_default_llama70b"}),
}
GRADERS = {}
for _c,(_i,_s,_l,gm) in CONDITIONS.items():
    for g,r in gm.items(): GRADERS.setdefault(g, {}).update(load_grader(r))

def auroc(s,y):
    s=np.asarray(s,float); y=np.asarray(y,bool)
    pos=int(y.sum()); neg=y.size-pos
    if pos==0 or neg==0: return None
    o=np.argsort(s,kind="mergesort"); ss=s[o]; r=np.empty(s.size,float); i=0
    while i<ss.size:
        j=i
        while j+1<ss.size and ss[j+1]==ss[i]: j+=1
        r[o[i:j+1]]=(i+j)/2+1; i=j+1
    return float((r[y].sum()-pos*(pos+1)/2)/(pos*neg))

def aurac(s,c):
    s=np.asarray(s,float); c=np.asarray(c,bool)[np.argsort(s,kind="mergesort")]
    n=c.size; cum=np.cumsum(c); qs=np.linspace(0.1,1.0,20)
    a=np.array([cum[max(1,int(round(q*n)))-1]/max(1,int(round(q*n))) for q in qs])
    return float(a.sum()*(qs[1]-qs[0])), float(np.mean(cum/np.arange(1,n+1)))

def answer_of(r):
    a = r.get("most_likely_answer") or ""
    return a.get("response","") if isinstance(a,dict) else a

def load_cell(path, grader):
    k = key(Path(path).parent.name)
    jl = GRADERS.get(grader,{}).get(k,{}) if grader.startswith("llm_") else {}
    if grader.startswith("llm_") and not jl: return k, []
    recs, seen = [], set()
    for line in open(path):
        r = json.loads(line)
        if k and k[2]=="triviaqa" and k[1]=="eval":
            p=r["prompt"]
            if p in EXCL or p in seen: continue
            seen.add(p)
        if jl:
            pid=r.get("prompt_id")
            if pid not in jl: continue
            r["correctness_label"]=jl[pid]
        elif grader=="squad_token_f1" and "correctness_label_squad" in r:
            r["correctness_label"]=r["correctness_label_squad"]   # E2 fix
        recs.append(r)
    return k, recs

rows=[]
def emit(arm, condition, instruction, shots, backend, grader, path, methods, temp="1.0"):
    k, recs = load_cell(path, grader)
    if not recs or not k: return
    correct=[bool(r["correctness_label"]) for r in recs]
    y=[not c for c in correct]
    L=[len(answer_of(r)) for r in recs]
    for m in methods:
        if m not in recs[0].get("scores",{}): continue
        sc=[r["scores"][m] for r in recs]
        a=auroc(sc,y)
        if a is None: continue
        ap,am=aurac(sc,correct)
        rows.append(dict(arm=arm, condition=condition, instruction=instruction,
            n_demonstrations=shots, split_role=k[1], dataset=k[2], model=k[3],
            seed=k[4], temperature=temp, entailment_backend=backend, grader=grader,
            method=m, n_records=len(recs), mean_answer_chars=round(st.mean(L),2),
            accuracy=round(sum(correct)/len(correct),6), auroc=round(a,6),
            aurac_paper=round(ap,6), aurac_mean_retained=round(am,6)))

def graders_for(cond):
    return ["squad_token_f1"]+list(CONDITIONS[cond][3])

# ---- short answer: four conditions x every backend ----
for cond,(instr,shots,lroot,_g) in CONDITIONS.items():
    for bdir in sorted(Path(lroot).iterdir()) if Path(lroot).is_dir() else []:
        if not bdir.is_dir(): continue
        for f in sorted(bdir.glob(f"repl-{instr}-eval-*/scored.jsonl")):
            for g in graders_for(cond):
                emit("short_answer",cond,instr,shots,bdir.name,g,f,ENTROPY)

# ---- P(True) and probes (original two conditions only) ----
for src,methods,backend in (
    ("results/ptrue",["ptrue_uncertainty"],"microsoft_deberta-v2-xlarge-mnli"),
    ("results/probe_scored",["probe_uncertainty","accuracy_probe_uncertainty"],
     "cross-encoder_nli-deberta-v3-large")):
    for f in sorted(Path(src).glob("*/scored.jsonl")):
        k=key(f.parent.name)
        if not k: continue
        cond = "chat_0shot" if k[0]=="chat" else "default_5shot"
        instr,shots,_,_ = CONDITIONS[cond]
        for g in graders_for(cond):
            emit("short_answer",cond,instr,shots,backend,g,f,methods)

# ---- temperature sweep, both backends ----
for t in ("0.1","0.3","0.5","0.7"):
    root=Path(f"results/ladder_T{t}")
    for bdir in sorted(root.iterdir()) if root.is_dir() else []:
        if not bdir.is_dir(): continue
        for f in sorted(bdir.glob("repl-*-eval-*/scored.jsonl")):
            k=key(f.parent.name)
            if not k: continue
            cond = "chat_0shot" if k[0]=="chat" else "default_5shot"
            instr,shots,_,_ = CONDITIONS[cond]
            for g in graders_for(cond):
                emit("short_answer",cond,instr,shots,bdir.name,g,f,ENTROPY,temp=t)

# ---- long form ----
bio=Path("results_sep_v3_bio/tables/metric_summary.csv")
if bio.is_file():
    for r in csv.DictReader(open(bio)):
        try: a=float(r["auroc"])
        except (ValueError,TypeError,KeyError): continue
        rows.append(dict(arm="long_form", condition="paragraph", instruction="bio",
            n_demonstrations=0, split_role=r.get("split",""), dataset="bio",
            model=r.get("model",""), seed=r.get("seed",""), temperature="1.0",
            entailment_backend=r.get("clustering",""), grader="llm_qwen2.5-72b",
            method=r.get("method",""), n_records=r.get("num_records",""),
            mean_answer_chars="", accuracy=r.get("raw_accuracy",""),
            auroc=round(a,6), aurac_paper=r.get("aurac",""), aurac_mean_retained=""))

F=["arm","condition","instruction","n_demonstrations","split_role","dataset","model",
   "seed","temperature","entailment_backend","grader","method","n_records",
   "mean_answer_chars","accuracy","auroc","aurac_paper","aurac_mean_retained"]
with open(OUT/"measurements.csv","w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=F); w.writeheader(); w.writerows(rows)
print(f"measurements.csv      {len(rows)} rows")

# ---- grader agreement, deduplicated ----
lr=[]
q,l = GRADERS.get("llm_qwen2.5-72b",{}), GRADERS.get("llm_llama-3.1-70b",{})
for k in sorted(set(q) & set(l)):
    ids=set(q[k]) & set(l[k])
    if not ids: continue
    ag=sum(q[k][i]==l[k][i] for i in ids)
    lr.append(dict(regime=k[0],split_role=k[1],dataset=k[2],model=k[3],seed=k[4],
        n_records=len(ids),
        accuracy_qwen=round(sum(q[k][i] for i in ids)/len(ids),6),
        accuracy_llama=round(sum(l[k][i] for i in ids)/len(ids),6),
        agreement=round(ag/len(ids),6)))
with open(OUT/"label_agreement.csv","w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(lr[0])); w.writeheader(); w.writerows(lr)
print(f"label_agreement.csv   {len(lr)} rows")

# ---- cluster counts and backend agreement, deduplicated ----
counts=[]; clusters={}
for cond,(instr,shots,lroot,_g) in CONDITIONS.items():
    for bdir in sorted(Path(lroot).iterdir()) if Path(lroot).is_dir() else []:
        if not bdir.is_dir(): continue
        per=[]
        for f in sorted(bdir.glob(f"repl-{instr}-eval-*/scored.jsonl")):
            _k,recs=load_cell(f,"squad_token_f1")
            for r in recs:
                n=len(set(r["semantic_clusters"])); per.append(n)
                clusters.setdefault(bdir.name,{})[(f.parent.name,r["prompt_id"])]=n
        if per: counts.append(dict(condition=cond,entailment_backend=bdir.name,
            mean_clusters_per_record=round(float(np.mean(per)),4),n_records=len(per)))
with open(OUT/"cluster_counts.csv","w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(counts[0])); w.writeheader(); w.writerows(counts)
print(f"cluster_counts.csv    {len(counts)} rows")

names=sorted(clusters); ar=[]
for i in range(len(names)):
    for j in range(i+1,len(names)):
        a,b=clusters[names[i]],clusters[names[j]]
        both=set(a)&set(b)
        if not both: continue
        ar.append(dict(backend_a=names[i],backend_b=names[j],n_records=len(both),
            identical_cluster_count=round(sum(a[k]==b[k] for k in both)/len(both),6)))
with open(OUT/"backend_agreement.csv","w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(ar[0])); w.writeheader(); w.writerows(ar)
print(f"backend_agreement.csv {len(ar)} rows")

# ---- seed variability ----
grp=defaultdict(list)
for r in rows:
    if r["arm"]!="short_answer" or r["temperature"]!="1.0": continue
    grp[(r["condition"],r["entailment_backend"],r["grader"],r["method"],
         r["dataset"],r["model"])].append(r["auroc"])
sv=defaultdict(list)
for (c,b,g,m,_d,_mo),v in grp.items():
    if len(v)>1: sv[(c,b,g,m)].append(max(v)-min(v))
out=[dict(condition=c,entailment_backend=b,grader=g,method=m,n_configurations=len(v),
     mean_seed_range=round(float(np.mean(v)),6),max_seed_range=round(float(np.max(v)),6))
     for (c,b,g,m),v in sorted(sv.items())]
with open(OUT/"seed_variability.csv","w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
print(f"seed_variability.csv  {len(out)} rows")

shutil.copy2("results/dedup_triviaqa.json", OUT/"dedup_triviaqa.json")
print("dedup_triviaqa.json   copied")
