# Detecting Hallucinations in Large Language Models Using Semantic Entropy

This repository contains thesis source, planning notes, and the first scaffold for experiments on hallucination detection with semantic entropy. The initial goal is a buildable LaTeX thesis project plus a clean place for future research code.

Current research direction: a two-phase experiment strategy. Phase 1 builds a small, sentence-length semantic entropy pipeline on local hardware and NHR@FAU Alex. Phase 2 scales the same design on Alex and Helma for larger models and stronger semantic judges. High-compute extensions such as Semantic Entropy Probes, Kernel Language Entropy, Semantic Energy, and Semantic Volume are optional Phase 2 work after the discrete baseline is stable.

Current status: the repository is organized, the LaTeX thesis builds locally, Chapters 1--9 have first-pass drafts or placeholders, and the stdlib-only Phase 1 scaffold is implemented for stub datasets, synthetic sampling, exact-match clustering, entropy scoring, metrics, plotting artifacts, and Slurm smoke testing. The Alex/A40 smoke job passed on 2026-05-17; the next step is a real 7B/8B model run on the Alex A100 partition. Results, discussion, and conclusion claims remain conditional until recorded real-model runs exist.

## Repository Layout

- `thesis/`: LaTeX thesis source, local kaobook styles, logos, figures, bibliography, chapters, and appendices.
- `docs/`: working thesis and experiment notes.
- `external/`: preserved uploaded papers, source notes, and original template material.
- `src/`: Python package (`semantic_entropy`) with the Phase 1 experiment scaffold.
- `configs/`: experiment configuration files.
- `slurm/`: HPC Slurm job scripts for Alex and Helma.
- `data/`: local datasets and processed data placeholders.
- `results/`: figures, tables, and small intentional results.
- `tests/`: Python tests for the `semantic_entropy` package.

## Local LaTeX Build

From the repository root:

```sh
cd thesis
latexmk -pdf main.tex
```

`thesis/latexmkrc` currently uses `pdflatex` with nonstop mode and halt-on-error. It also adds `thesis/styles/` to `TEXINPUTS` so `main.tex` can find the local kaobook class and style files.

Current local toolchain: Homebrew TeX Live 2026 provides `latexmk`, `pdflatex`, `bibtex`, and `biber`. The thesis currently builds with `latexmk -pdf main.tex`; known build status is tracked in `MANIFEST_REORG.md`.

## Planned Experiment Workflow

Phase 1: local and Alex pilot

1. Use TriviaQA and SVAMP as the first datasets.
2. Run a small local model first, then a 7B/8B instruct or chat model on Alex/A100.
3. Sample `M = 4` answers for smoke tests and `M = 10` for the first meaningful comparison.
4. Compare surface-form uncertainty, naive answer diversity, and discrete semantic entropy.
5. Cluster sampled answers with a local NLI backend before using larger judges.
6. Evaluate AUROC, AURAC, rejection-accuracy curves, and raw accuracy as context.

Phase 2: NHR@FAU HPC scaling

1. Use Alex for scaled inference and robust NLI clustering on A100-class GPUs.
2. Use Helma for 70B+ models and high-cost semantic judges; reserve Semantic Entropy Probes and other high-compute extensions for optional runs after the baseline is stable.
3. Optionally expand from TriviaQA/SVAMP to SQuAD, BioASQ, and NQ-Open.
4. Keep Slurm job IDs, commands, GPU type, runtime, memory, and output paths with every run.
5. Keep large model caches, raw outputs, checkpoints, and downloaded datasets out of git.

## Smoke Commands

All commands below write to `/tmp` so they do not create tracked result files.
The all-correct fixture exercises scoring, metrics export, table export, and
plotting end to end. Its curves are expected to be flat and AUROC is expected
to be `null`, because both records are correct.

```sh
rm -rf /tmp/se_smoke_metrics /tmp/se_smoke_tables /tmp/se_smoke_figures

PYTHONPATH=src python3 -m semantic_entropy.cli \
  tests/fixtures/synthetic_unscored.jsonl \
  /tmp/se_smoke_scored.jsonl

PYTHONPATH=src python3 -m semantic_entropy.metrics \
  /tmp/se_smoke_scored.jsonl \
  --output-dir /tmp/se_smoke_metrics \
  --table-dir /tmp/se_smoke_tables \
  --table-basename smoke_metric_summary

uv run --extra plot python -m semantic_entropy.plotting \
  /tmp/se_smoke_metrics/rejection_accuracy_curves.csv \
  --output-dir /tmp/se_smoke_figures \
  --basename smoke_rejection_accuracy
```

The mixed-correctness fixture is already scored. Use it when the goal is a
visually meaningful metrics/plot sanity check with non-flat rejection-accuracy
curves.

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

## Research Corpus

- Primary note: `docs/reading_notes/nature_semantic_entropy.md`.
- Expanded literature map: `docs/reading_notes/literature_review.md`.
- Chapter 2 background notes: `docs/reading_notes/chapter_02_background_notes.md`.
- Writing integrity guide: `docs/writing_integrity.md`.
- Local PDF index: `external/papers/README.md`.
- Bibliography: `thesis/main.bib`.

The current corpus includes the Nature semantic entropy paper, the ICLR semantic uncertainty precursor, Semantic Entropy Probes, Kernel Language Entropy, SelfCheckGPT, P(True), hallucination/UQ surveys, dataset citations, model citations, and newer related work on semantic energy, semantic volume, high-certainty hallucinations, and neuron/probe-based detection.

Thesis prose should follow the writing integrity guide before final plagiarism and AI-detection review. Detector output should be treated as a review queue, not as a writing objective.

## Data and Artifacts

Do not commit large data, raw datasets, model outputs, checkpoints, downloaded models, or bulk generated artifacts. Keep placeholders such as `data/raw/README.md` and `results/README.md` tracked so the intended layout remains visible.
