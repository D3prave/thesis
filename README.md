# Detecting Hallucinations in Large Language Models Using Semantic Entropy

This repository contains the LaTeX source, Python research code, experiment
configuration, and recorded small artifacts for a bachelor's thesis on semantic
entropy as a training-free hallucination detection and selective abstention
method for short-answer question answering.

The central question is whether entropy over meanings separates likely
incorrect answers from correct answers better than entropy over surface forms.
The thesis studies this on TriviaQA and SVAMP using sampled answers from
Mistral-7B-Instruct-v0.3 and Llama-3.1-70B-Instruct.

## Thesis Status

The thesis is complete. All nine chapters, the abstract, the
acknowledgements, the appendix, the bibliography, and all figures are
written, and the PDF builds cleanly with `latexmk -pdf main.tex`. Phase~1
evaluates exact-match clustering after answer normalization. Phase~2 repeats
the same model--dataset grid with bidirectional Natural Language Inference
clustering using DeBERTa-v3-base. Posthoc sensitivity checks on the Fritz
cluster compare DeBERTa-v3-large reclustering and Kernel Language Entropy.
Semantic Entropy Probes (SEP) are scaffolded but were not run within the
thesis scope; see Chapter~7 and `PLANS.md` for the rationale and the
reopen-recipe.

Key results:

- Llama-3.1-70B benefits from discrete semantic entropy over naive sample
  entropy: AUROC 0.696 vs. 0.644 on TriviaQA and 0.842 vs. 0.827 on SVAMP.
- Mistral-7B shows little or no gain from semantic clustering, suggesting that
  the method is most useful when the generator is already reasonably
  consistent.
- NLI clustering is mixed: it improves Mistral-7B on TriviaQA by 1.3 AUROC
  points but slightly reduces Llama-3.1-70B on TriviaQA by 1.8 points.
- Rejection-accuracy curves increase across the evaluated conditions, so
  sampling-based entropy is useful for selective abstention even when absolute
  accuracy differs by model and dataset.
- Posthoc DeBERTa-v3-large reclustering partially recovers the 70B/TriviaQA
  regression but slightly worsens 7B/TriviaQA, so a stronger NLI judge alone
  does not flip the qualitative story.
- Kernel Language Entropy with the tested sentence encoder
  (`all-MiniLM-L6-v2`) does not outperform discrete semantic entropy on any
  of the four posthoc runs.

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

To regenerate the tracked aggregate summary tables and figures from recorded
run directories:

```sh
uv run --extra plot python scripts/aggregate_results.py --plot
```

## Data and Artifacts

Large datasets, model weights, raw model outputs, and bulk generated artifacts
are intentionally kept out of git. Small fixtures, summary tables, and selected
figures are tracked when they support thesis reproducibility.

The recorded metric summary is available at
`results/tables/metric_summary.csv` (rows are tagged by
`clustering = exact-match`, `nli`, `posthoc-kle`, or
`posthoc-nli-deberta-v3-large`). Generated figures are under
`results/figures/`; the figures used in the thesis are committed under
`thesis/figures/`.
