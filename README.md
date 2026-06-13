# Detecting Hallucinations in Large Language Models Using Semantic Entropy

This repository contains the LaTeX source, Python research code, experiment
configuration, and recorded small artifacts for a bachelor's thesis on semantic
entropy as a training-free hallucination detection and selective abstention
method for short-answer question answering and long-form biography generation.

The central question is whether entropy over meanings separates likely
incorrect answers from correct answers better than entropy over surface forms.
The thesis studies this on TriviaQA, NQ-Open, SVAMP, and a FactScore-style
biography task using sampled answers from Mistral-7B-Instruct-v0.3,
Llama-3.1-8B-Instruct, and Llama-3.1-70B-Instruct (3 models × 3 seeds).

## Thesis Status

**As of 2026-06-13 — 3-seed rerun in progress; bio rerun pending.**

A vLLM seed-propagation bug (the `seed` parameter was accepted but never
forwarded to `vllm.LLM()`) was discovered during a full scientific validity
audit. All "seeded" runs prior to this fix silently used seed=0, making the
three nominal seeds effectively identical. The bug is fixed and committed
(`1bd9026`). 27 short-form generation jobs (Slurm IDs 3739810–3739836) are
running on the NHR@FAU Alex cluster with correct seed propagation and
paper-matching correctness labelling (modal-answer + SQuAD-F1 > 0.5).

A bio system-prompt bug was also found: all bio generations used the terse QA
prompt ("answer in a word or phrase"), producing 9–12 word responses instead
of 120–150 word paragraphs. Fixed in all bio sbatch files; 9 bio generation
jobs will be submitted after the short-form rerun completes.

**What is currently valid (no rerun needed):**
- FactualBio AUROC (NLI-clustered, claim-level): Mistral-7B 0.688±0.004,
  Llama-3.1-8B 0.644±0.007, Llama-3.1-70B 0.521±0.001 — unaffected by
  both bugs.
- All metric formulas, clustering logic, and answer normalizers — verified
  correct in the 2026-06-13 audit (`SCIENTIFIC_VALIDITY_AUDIT_2026-06-13.md`).

Short-form AUROC numbers will be updated once the cluster jobs finish and
posthoc NLI reclustering has been rerun. The thesis draft is otherwise
structurally complete (nine chapters, abstract, bibliography, figures build
cleanly).

The surface-form baselines use raw-string entropy and exact-match clustering
after answer normalization. The main replication condition repeats the same
model–dataset grid with bidirectional Natural Language Inference clustering
using DeBERTa-v3-base. Additional analyses compare DeBERTa-v3-large
reclustering, Qwen2.5-72B LLM-judge reclustering, and Kernel Language Entropy,
plus held-out Semantic Entropy Probe (SEP) checks across both the short-answer
and biography tasks.

**Previously recorded key results (single-seed, any-sample labelling —
numbers will be superseded by the 3-seed rerun):**

- NLI clustering is mixed on short-answer tasks: it nominally raises Mistral-7B
  on TriviaQA but slightly reduces Llama-3.1-70B; neither difference is
  statistically significant.
- The long-form biography study shows the clearest evidence for semantic rather
  than surface uncertainty. Naive and surface entropy collapse to chance because
  paragraph samples are almost always unique. Meaning-aware methods stay useful.
- FactualBio AUROC (citable): Mistral-7B 0.688, Llama-3.1-8B 0.644,
  Llama-3.1-70B 0.521.
- SEP separates clearly from the chance-level surface baseline for both
  generators on biography.
- The current reproducibility verifier recomputes saved AUROC/AURAC values to
  within 1e-6.

The resulting thesis claim is scoped to semantic uncertainty and
confabulation-like errors. It is not a general factuality guarantee and does
not target high-confidence systematic errors.

## Repository Layout

- `thesis/`: LaTeX thesis source, bibliography, chapters, appendices, figures,
  and local thesis style files.
- `src/semantic_entropy/`: Python package for sampling records, clustering
  answers, computing entropy scores, metrics, plotting, and optional extension
  methods.
- `scripts/`: data preparation, aggregation, post-hoc scoring, and probe
  training helpers.
- `configs/`: experiment configuration files.
- `slurm/`: Slurm job scripts used for NHR@FAU runs.
- `docs/`: research notes, protocol notes, and literature notes.
- `data/`: small tracked fixture datasets and placeholders for local data.
- `results/`: recorded summary tables and intentional figures.
- `tests/`: Python tests and synthetic JSONL fixtures.
- `external/`: preserved source papers, notes, and original thesis template
  material.

## Build the Thesis

From the repository root:

```sh
cd thesis
latexmk -pdf main.tex
```

Generated LaTeX build products, including `thesis/main.pdf`, are ignored by
git.

## Run the Python Checks

```sh
uv run pytest tests/ -q
```

The package keeps heavyweight model dependencies optional. Install only the
extras needed for the workflow being reproduced.

## Reproduce Small Smoke Outputs

These commands write to `/tmp` and do not create tracked result files.

```sh
rm -rf /tmp/se_mixed_metrics /tmp/se_mixed_tables /tmp/se_mixed_figures

PYTHONPATH=src python3 -m semantic_entropy.metrics \
  tests/fixtures/synthetic_mixed_scored.jsonl \
  --output-dir /tmp/se_mixed_metrics \
  --table-dir /tmp/se_mixed_tables \
  --table-basename mixed_metric_summary

uv run --extra plot python -m semantic_entropy.plotting \
  /tmp/se_mixed_metrics/rejection_accuracy_curves.csv \
  --output-dir /tmp/se_mixed_figures \
  --basename mixed_rejection_accuracy
```

To regenerate the aggregate summary tables and canonical figures from recorded
run directories:

```sh
uv run --extra plot python scripts/aggregate_results.py --plot
uv run python scripts/build_canonical_analysis.py
```

## Biography Post-Processing

The paragraph-length biography DeBERTa-large and `nli-roberta-large` KLE
sensitivity jobs have completed. After syncing biography posthoc outputs from
the cluster, write durable labeled artifacts and refresh the aggregate metrics:

```sh
bash scripts/finalize_bio_study.sh
```

The recorded biography posthoc rows above are fixed-label sensitivity checks:
they reuse labels from an earlier modal representative. To rerun correctness
for each backend's own modal representative, first inspect the discovered
inputs, then submit on Alex:

```sh
bash scripts/dispatch_bio_backend_regrade_alex.sh --dry-run
bash scripts/dispatch_bio_backend_regrade_alex.sh --submit
```

Those jobs write under `results/phase2/` and set
`bio_regrade_source_clustering` in `job_meta.json`. Aggregation labels them as
`bio-regrade-<source-clustering>` so they cannot be confused with the
fixed-label posthoc rows.

## FactualBio-Style Extension

For a paper-style long-form follow-up, prepare deterministic prompt JSONL files
from existing biography outputs, run the expensive LLM stages on the cluster,
then score one factual claim per record:

```sh
python scripts/prepare_factual_bio_prompts.py extract-claims \
  results/phase1/<bio-run>/scored.jsonl /tmp/factualbio_claim_prompts.jsonl

python scripts/prepare_factual_bio_prompts.py questions \
  /tmp/factualbio_claim_records.jsonl /tmp/factualbio_question_prompts.jsonl

python scripts/prepare_factual_bio_prompts.py answers \
  /tmp/factualbio_claim_records.with_questions.jsonl /tmp/factualbio_answer_prompts.jsonl

python scripts/score_factual_bio_claims.py \
  /tmp/factualbio_claim_records.with_answers.jsonl \
  /tmp/factualbio_claim_records.scored.jsonl
```

The scorer computes discrete semantic entropy per generated question over the
expected answer plus regenerated short answers, averages question entropies to
a claim score, and writes claim-level AUROC/AURAC artifacts.

## Data and Artifacts

Large datasets, model weights, raw model outputs, generated result summaries,
and generated figures are intentionally kept out of git for now. Small fixtures
remain tracked when they support local checks.

The locally generated metric summary is written to
`results/tables/metric_summary.csv` (rows are tagged by
`clustering = exact-match`, `nli`, `sep`, `posthoc-kle-all-minilm-l6-v2`,
`posthoc-kle-nli-roberta-large`, `posthoc-nli-deberta-v3-large`, or
`posthoc-llm-judge-qwen2.5-72b-instruct`; backend-specific biography regrades
use `bio-regrade-*`). Generated figures are written under `results/figures/`
and copied into `thesis/figures/` when needed for local LaTeX builds.
