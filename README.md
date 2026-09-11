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
reimplementation on a smaller recorded grid, comparisons of estimators,
entailment models and graders, and a decoding-temperature sweep. A separate
whole-paragraph experiment is exploratory.

## What it found

On sentence-length answers, discrete semantic entropy improves on the
likelihood-based naive baseline, but its average gain over normalized surface
entropy is smaller and depends on the entailment model and grader. On short
phrases, the count-based estimators are close. These are findings about the
recorded models and tasks, not a general ranking of all detectors.

The thesis reports the full recorded comparisons in its Results chapter and
supporting appendix. It does not use human correctness labels. The long-form
experiment measures paragraph acceptance against a saved reference, using a
different grouping protocol from the original paragraph experiment. It is
exploratory rather than a central replication result.

## Layout

| Path | Contents |
|---|---|
| `thesis/` | LaTeX source: 7 chapters, appendix, bibliography, figures |
| `thesis/includes/` | 23 active generated table files; some contain several tables |
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
`biblatex`/`biber`). The output is `thesis/main.pdf`.

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

The active result tables are generated from the committed measurement package,
auxiliary summaries, and paired comparisons. The shared builder also reads
archived cross-dataset probe exports for inactive tables; those transfer
results are outside the thesis scope.
The following checks validate the expected coverage and registered producers;
they do not certify complete raw-data provenance:

```sh
python3 scripts/check_thesis_measurements.py
python3 scripts/check_table_provenance.py
```

Regenerate the active tables and analysis figures locally, without model
weights or new cluster runs:

```sh
python3 scripts/build_thesis_tables.py \
  docs/independent_analysis/data/measurements.csv thesis/includes
```

The builder requires NumPy, pandas, and Matplotlib. The current cluster exporter
is `scripts/export_data_package_v2.py`; it needs the original result trees.
See `docs/independent_analysis/data/README.md` for coverage and field meanings,
and `docs/cluster_coverage_audit_2026-09-05.md` for the raw-artifact audit.
Older builders and results remain in the repository but do not supply the
active thesis tables.

For source packaging, artifact hashes, recovered protocol metadata, and the
limits of reproducing historical runs, see `docs/reproduction.md` and
`docs/reproducibility_manifest.json`. The September 11 correction register is
`docs/audit_resolution_2026-09-11.md`.

For the editable thesis handoff, see `docs/submission_checklist.md`. The final
result audit is in `docs/final_consistency_audit_2026-09-05.md`; the literature
claim checks and their access limits are in
`docs/reading_notes/source_claim_audit_2026-09-05.md`.

## Published FactualBio labels

The repository retains an importer for the human-annotated claims from the code
release accompanying Farquhar et al. These labels are not used in the current
thesis results. The source is pinned by commit and file hash:

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

The main comparison is discrete semantic entropy against normalized surface
entropy. Statistical summaries keep the three seeds nested within each
dataset–generator pair. The thesis reports uncertainty in those comparisons
and does not treat seeds as independent datasets.

Correctness is assessed by token-F1 and two LLM graders for short answers, and
by Qwen2.5-72B-Instruct for biographies. There is no human validation of these
labels. The main temperature tables use only LLM labels of the fixed answer;
token-F1 sweep labels instead grade the modal sample and are not comparable.
Prompting and long-form analyses are secondary or exploratory. AURAC and
cross-dataset probe transfer remain in the research archive, outside the active
thesis results.

## Citation

Jakub Wiśniewski, *Detecting Hallucinations in Large Language Models Using
Semantic Entropy*. Bachelor's thesis, Katholische Universität
Eichstätt-Ingolstadt, 2026.
