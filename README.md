# Detecting Hallucinations in Large Language Models Using Semantic Entropy

This repository contains thesis source, planning notes, and the first scaffold for experiments on hallucination detection with semantic entropy. The initial goal is a buildable LaTeX thesis project plus a clean place for future research code.

Current research direction: reproduce a small, sentence-length semantic entropy pipeline first, then decide whether Semantic Entropy Probes or other efficient variants are realistic extensions.

## Repository Layout

- `thesis/`: LaTeX thesis source, local kaobook styles, logos, figures, bibliography, chapters, and appendices.
- `docs/`: working thesis and experiment notes.
- `external/`: preserved uploaded papers, source notes, and original template material.
- `src/`: Python package and scripts for future experiments.
- `configs/`: future experiment configuration files.
- `slurm/`: future HPC job scripts.
- `data/`: local datasets and processed data placeholders.
- `results/`: figures, tables, and small intentional results.
- `tests/`: future Python tests.

## Local LaTeX Build

From the repository root:

```sh
cd thesis
latexmk -pdf main.tex
```

`thesis/latexmkrc` currently uses `pdflatex` with nonstop mode and halt-on-error. It also adds `thesis/styles/` to `TEXINPUTS` so `main.tex` can find the local kaobook class and style files.

Current local toolchain: Homebrew TeX Live 2026 provides `latexmk`, `pdflatex`, `bibtex`, and `biber`. The thesis currently builds with `latexmk -pdf main.tex`; known build status is tracked in `MANIFEST_REORG.md`.

## Planned Experiment Workflow

1. Read and summarize the semantic entropy paper and related uncertainty-estimation literature.
2. Run a small pilot on TriviaQA and SVAMP.
3. Sample multiple answers per prompt with `M = 4` and `M = 10`.
4. Compare naive entropy, discrete semantic entropy, and later full semantic entropy.
5. Cluster sampled answers with a local NLI model first.
6. Evaluate AUROC, AURAC, and rejection-accuracy curves.
7. Scale carefully on HPC with Slurm after debug jobs pass locally.

## Research Corpus

- Primary note: `docs/reading_notes/nature_semantic_entropy.md`.
- Expanded literature map: `docs/reading_notes/literature_review.md`.
- Chapter 2 background notes: `docs/reading_notes/chapter_02_background_notes.md`.
- Local PDF index: `external/papers/README.md`.
- Bibliography: `thesis/main.bib`.

The current corpus includes the Nature semantic entropy paper, the ICLR semantic uncertainty precursor, Semantic Entropy Probes, Kernel Language Entropy, SelfCheckGPT, P(True), hallucination/UQ surveys, dataset citations, model citations, and newer related work on semantic energy, semantic volume, high-certainty hallucinations, and neuron/probe-based detection.

## Data and Artifacts

Do not commit large data, raw datasets, model outputs, checkpoints, downloaded models, or bulk generated artifacts. Keep placeholders such as `data/raw/README.md` and `results/README.md` tracked so the intended layout remains visible.
