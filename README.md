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

Complete draft as of 2026-07-02 (seven chapters, replication-first structure).
Short-answer results cover TriviaQA, NQ-Open, and SVAMP across three models
and three seeds, plus a temperature sweep. The long-form extension applies
the same detectors to FactScore-style biography generation. Canonical results
are locked and verified against raw records; see the thesis PDF for figures,
tables, and discussion.

## Repository Layout

- `thesis/`: LaTeX source (chapters, bibliography, figures, styles).
- `src/semantic_entropy/`: Python package — sampling, clustering, entropy
  scoring, metrics, plotting.
- `scripts/`: data prep, aggregation, and post-hoc scoring helpers.
- `slurm/`: Slurm job scripts used for NHR@FAU cluster runs.
- `tests/`: test suite with synthetic fixtures.
- `data/`: small tracked fixture datasets.

Raw results, generated figures, and internal working notes are kept out of
git and stay local.

## Build the Thesis

```sh
cd thesis && latexmk -pdf main.tex
```

## Run the Tests

```sh
uv run pytest tests/ -q
```
