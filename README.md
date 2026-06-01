# Detecting Hallucinations in Large Language Models Using Semantic Entropy

This repository contains the LaTeX source, Python research code, experiment
configuration, and recorded small artifacts for a bachelor's thesis on semantic
entropy as a training-free hallucination detection and selective abstention
method for short-answer question answering and long-form biography generation.

The central question is whether entropy over meanings separates likely
incorrect answers from correct answers better than entropy over surface forms.
The thesis studies this on TriviaQA, SVAMP, and a FactScore-style biography
task using sampled answers from Mistral-7B-Instruct-v0.3 and
Llama-3.1-70B-Instruct.

## Thesis Status

The thesis draft is complete apart from final administrative fields such as
the submission date. All nine chapters, the abstract, the acknowledgements,
the appendix, the bibliography, and all figures are written, and the PDF
builds cleanly with `latexmk -pdf main.tex`. The surface-form baselines use
raw-string entropy and exact-match clustering after answer normalization. The
main replication condition repeats the same model--dataset grid with
bidirectional Natural Language Inference clustering using DeBERTa-v3-base.
Additional analyses compare
DeBERTa-v3-large reclustering, Qwen2.5-72B LLM-judge reclustering, and
Kernel Language Entropy, plus held-out Semantic Entropy Probe (SEP)
checks for both generators across the short-answer and biography tasks.

Key results:

- In the exact-match baseline, Llama-3.1-70B's normalized
  surface-cluster score ranks errors better than naive sample entropy: AUROC
  0.696 vs. 0.644 on TriviaQA and 0.842 vs. 0.827 on SVAMP. This is not, by
  itself, evidence for true semantic entropy because exact match does not use NLI
  semantic clustering.
- On the short-answer tasks, Mistral-7B shows little or no gain from semantic
  clustering. Biography behaves differently because surface matching fails on
  paragraph samples.
- NLI clustering is mixed: it nominally raises Mistral-7B on TriviaQA by
  1.3 AUROC points but slightly reduces Llama-3.1-70B on TriviaQA by
  1.8 points; neither difference is statistically significant in the recorded
  bootstrap check.
- Rejection-accuracy curves generally trend upward across the evaluated
  conditions, so sampling-based entropy is useful for selective abstention even
  when absolute accuracy differs by model and dataset. The curves are not
  monotonic: local drops occur when a rejected high-uncertainty record was
  actually correct, and the final few retained records are unstable. Additional
  thesis plots stop at 95% rejection to exclude this right-tail instability.
- Posthoc DeBERTa-v3-large reclustering partially recovers the 70B/TriviaQA
  regression but slightly worsens 7B/TriviaQA, so a stronger NLI judge alone
  does not flip the short-answer story. On biography it has a material effect:
  AUROC rises from 0.643 to 0.732 for Mistral-7B and from 0.454 to 0.656 for
  Llama-3.1-70B. These biography numbers are fixed-label reclustering
  sensitivity checks, not fresh end-to-end correctness re-grades.
- On the four short-answer cells, Qwen2.5-72B LLM-judge reclustering remains
  below surface entropy, so judge scale alone is not sufficient in this setup.
  Biography is different: the same judge recovers a useful clustering signal.
- Kernel Language Entropy with `all-MiniLM-L6-v2` does not outperform
  discrete semantic entropy on any of the four posthoc runs. The
  `nli-roberta-large` KLE ablation improves the two Mistral-7B cells but
  does not uniformly close the gap on Llama-3.1-70B.
- The long-form biography study is the clearest evidence for *semantic*
  rather than surface uncertainty. Naive and surface entropy collapse to
  chance (paragraph samples are almost always unique) and KLE falls below
  chance, while semantic clustering and the hidden-state probe stay useful:
  SEP reaches AUROC 0.763 on Llama-3.1-70B and LLM-judge equivalence clustering
  reaches 0.805 on Mistral-7B.
- SEP should be compared with its same-split held-out baselines rather than
  treated as a direct rescore of the replication rows. On short answers its
  result is task-dependent for Mistral-7B and favorable but not statistically
  separated from zero for Llama-3.1-70B. On biography it separates clearly
  from the chance-level surface baseline for both generators.
- The 2026-06-01 reproducibility audit verified that every canonical run's
  AUROC/AURAC recomputes from `scored.labeled.jsonl` when present, otherwise
  `scored.jsonl`, to within 1e-6, and that the corrected sentence-pair DeBERTa
  reruns reproduce the pre-fix numbers; see `docs/results_audit_2026-06-01.md`.

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

## Data and Artifacts

Large datasets, model weights, raw model outputs, generated result summaries,
and generated figures are intentionally kept out of git for now. Small fixtures
remain tracked when they support local checks.

The locally generated metric summary is written to
`results/tables/metric_summary.csv` (rows are tagged by
`clustering = exact-match`, `nli`, `sep`, `posthoc-kle-all-minilm-l6-v2`,
`posthoc-kle-nli-roberta-large`, `posthoc-nli-deberta-v3-large`, or
`posthoc-llm-judge-qwen2.5-72b-instruct`). Generated figures are written under
`results/figures/` and copied into `thesis/figures/` when needed for local
LaTeX builds.
