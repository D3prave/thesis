# Detecting Hallucinations in Large Language Models Using Semantic Entropy

Bachelor's thesis, Mathematical Institute for Machine Learning and Data Science,
Katholische Universität Eichstätt-Ingolstadt.

This repository holds the LaTeX source, the Python implementation, the Slurm
batch scripts, and the generated result tables for a study of **semantic
entropy** as a training-free hallucination-detection and selective-abstention
signal.

The question is whether entropy over *meanings* separates incorrect answers
from correct ones better than entropy over *surface forms*. It is evaluated on
TriviaQA, NQ-Open, SVAMP and a FactScore-style biography task, with answers
sampled from Mistral-7B-Instruct-v0.3, Llama-3.1-8B-Instruct and
Llama-3.1-70B-Instruct, three seeds each (42/43/44).

The method itself is due to [Farquhar et al.
(2024)](https://doi.org/10.1038/s41586-024-07421-0); the contribution here is a
reproducible reimplementation on a smaller local grid, plus a claim-level
long-form extension and a decoding-temperature sweep the original does not
report.

## What it found

**On short answers, meaning-based clustering does not pay for itself.** All
nine model–dataset cells have a negative mean difference against a normalized
exact-match surface baseline, and the single Holm-significant result runs
*against* semantic entropy. Dataset normalization removes most surface
variation before any semantic step runs, leaving little for clustering to
merge.

**At claim level on long-form text, it does.** Scoring biographies one claim at
a time against published human labels, semantic entropy leads the surface
baseline in every cell:

| Generator | AUROC (NLI) | AUROC (surface) | Δ |
|---|---|---|---|
| Mistral-7B | 0.748 | 0.642 | +0.105 |
| Llama-3.1-8B | 0.724 | 0.650 | +0.073 |
| Llama-3.1-70B | 0.701 | 0.619 | +0.082 |

The whole-paragraph comparison looks more favourable still, but is a diagnostic
rather than a fair contest: paragraphs almost never repeat as identical
strings, so the surface baseline sits pinned at its ceiling near chance.

**Two effects qualify practical use.** The entailment backend is immaterial on
short answers (four backends span ≤ 0.057 AUROC) and result-determining on
whole paragraphs, where a weak backend can drive the score below chance. And
sampling-based entropy strengthens with decoding temperature while the
single-pass P(True) baseline stays nearly flat, so which detector wins depends
on the decoding settings.

## Layout

| Path | Contents |
|---|---|
| `thesis/` | LaTeX source: 7 chapters, appendix, bibliography, figures |
| `thesis/includes/` | The 17 generated result tables the document `\input`s |
| `src/semantic_entropy/` | Python package: sampling, clustering, entropy scoring, metrics, probes, plotting |
| `scripts/` | Data preparation, aggregation, table and figure builders, verification gates |
| `slurm/` | Batch scripts for the NHR@FAU runs |
| `tests/` | Test suite with synthetic fixtures |
| `data/` | Small tracked fixture datasets |

Raw generations, hidden states, model caches and the sealed result roots are
bulky and live in cluster project storage, not in git.

## Build the thesis

```sh
cd thesis && latexmk -pdf main.tex
```

Requires a full TeX Live (the document uses `kaobook`, `newpx`, `algorithm2e`,
`biblatex`/`biber`). Builds to 83 pages with no undefined references.

## Run the tests

```sh
uv run pytest tests/ -q
```

## Use the package

Python ≥ 3.11, managed with [`uv`](https://docs.astral.sh/uv/). Dependencies are
grouped as optional extras so a light install stays light:

```sh
uv sync                 # core
uv sync --extra test    # pytest, ruff
uv sync --extra plot    # matplotlib
uv sync --extra hf      # torch, transformers (generation and probes)
```

The full pipeline (`semantic-entropy-run-pipeline`) needs a GPU and model
weights, but the scoring and metrics path runs on the committed synthetic
fixture with no models at all:

```sh
export PYTHONPATH=src
python3 -m semantic_entropy.fixtures tests/fixtures/synthetic_sampled_only.jsonl norm.jsonl
python3 -m semantic_entropy.cli     norm.jsonl scored.jsonl
python3 -m semantic_entropy.metrics scored.jsonl
```

That normalizes the sampled answers, clusters them by exact match, computes the
entropy scores, and prints AUROC/AURAC — the same code path the cluster runs
use, with the NLI backend left at its default.

`pyproject.toml` also installs these as console scripts
(`semantic-entropy-score-jsonl`, `-metrics-jsonl`, `-normalize-fixture`,
`-run-pipeline`, `-aggregate-results`, `-plot-curves`).

## Reproducibility

Every number in the thesis traces to a named script reading a sealed artifact
root. `thesis/includes/generated_*.tex` are script-produced; none is written by
hand, and `scripts/check_table_provenance.py` enforces this:

```sh
PYTHONPATH=src python3 scripts/check_table_provenance.py   # 17 tables, one sealed family
PYTHONPATH=src python3 scripts/check_figure_table_freshness.py
```

Appendix A.8 of the thesis lists the SHA-256 digest of every manifest binding
those numbers to their artifacts. Each run directory records a `job_meta.json`
with the Slurm job id, host, model and tokenizer revisions, decoding
parameters, seed, backend and input path.

The table builders read the sealed `results_sep_v3_*` roots, which are not in
this repository, so they only run where those roots are present:

```sh
PYTHONPATH=src python3 scripts/build_thesis_tables_from_summary.py \
  --sep-v2-manifest            results_sep_v3_T1/tables/sep_v2_protocol_manifest.json \
  --sep-v2-completion-manifest results_sep_v3_T1/tables/sep_v2_artifact_manifest.json \
  --sep-v2-summary             results_sep_v3_T1/tables/multiseed_method_summary.csv \
  --factualbio-surface-arm     results/tables/factualbio_surface_arm.csv
```

All four flags are required. The three manifest flags fail closed — the builder
refuses to emit a table from an input it cannot prove is sealed.
`--factualbio-surface-arm` fails *open*: without it the seven-column FactualBio
table is silently replaced by a three-column one, dropping the surface arm and
the deltas that carry the claim-level result. Only `git diff thesis/includes/`
catches that, so check it after every rebuild.

## Published FactualBio labels

The claim-level evaluation uses the 150 human-annotated claims from the code
release accompanying Farquhar et al., pinned by commit and file hash:

```sh
curl -fL https://raw.githubusercontent.com/jlko/long_hallucinations/957b806033d0b769db34beb7ce34316bd498f373/data.py \
  -o data/raw/factualbio_data_v1.0.0.py
shasum -a 256 data/raw/factualbio_data_v1.0.0.py
# Expected: b2ae5bc7fb86bc60c2c8ec189a510edb1d553ae573b9157a8832056ecc81f543

PYTHONPATH=src python3 scripts/import_published_factualbio.py \
  data/raw/factualbio_data_v1.0.0.py \
  data/processed/factualbio_published_claims.jsonl \
  data/processed/factualbio_published_labels.jsonl
```

Importing does not execute the upstream file. The importer requires exactly 21
biographies, 150 claim annotations and the published 105 correct / 45 false
split before writing anything. `src/semantic_entropy/label_provenance.py` keeps
published human labels and locally generated LLM-judge labels distinct in a
machine-checkable way, so an automated judgment cannot be mistaken for a human
annotation.

## Scope and caveats

The detection target is *confabulation* — answers that are wrong and
inconsistent across resampling. High-confidence systematic errors, where the
model is wrong the same way every time, are outside it by construction.

Exactly one comparison is preregistered and confirmatory: discrete semantic
entropy against the normalized surface baseline across the nine short-answer
cells, with Holm correction over that nine-member family. Everything else —
clustering backends, probes, P(True), KLE, temperature, both long-form analyses
— is exploratory and reported as such.

Whole-paragraph biography correctness labels come from an open LLM judge
(Qwen2.5-72B-Instruct) against pinned Wikipedia references; the claim-level
evaluation uses human labels and is the stronger of the two.

## Citation

Jakub Wiśniewski, *Detecting Hallucinations in Large Language Models Using
Semantic Entropy*. Bachelor's thesis, Katholische Universität
Eichstätt-Ingolstadt, 2026.
