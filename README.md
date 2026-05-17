# Detecting Hallucinations in Large Language Models Using Semantic Entropy

This repository contains thesis source, planning notes, and the first scaffold for experiments on hallucination detection with semantic entropy. The initial goal is a buildable LaTeX thesis project plus a clean place for future research code.

Current research direction: a two-phase experiment strategy. Phase 1 builds a small, sentence-length semantic entropy pipeline on local hardware or TinyGPU. Phase 2 scales the same design on NHR@FAU Alex and Helma for larger models and stronger semantic judges. High-compute extensions such as Semantic Entropy Probes, Kernel Language Entropy, Semantic Energy, and Semantic Volume are optional Phase 2 work after the discrete baseline is stable.

Current status: the repository is organized, the LaTeX thesis builds locally, Chapters 1--9 have first-pass drafts or placeholders, and no Python experiment pipeline has been implemented yet. Results, discussion, and conclusion claims remain conditional until recorded runs exist.

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

Phase 1: local and TinyGPU pilot

1. Use TriviaQA and SVAMP as the first datasets.
2. Run a small local model first, then a 7B/8B instruct or chat model.
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

## Research Corpus

- Primary note: `docs/reading_notes/nature_semantic_entropy.md`.
- Expanded literature map: `docs/reading_notes/literature_review.md`.
- Chapter 2 background notes: `docs/reading_notes/chapter_02_background_notes.md`.
- Local PDF index: `external/papers/README.md`.
- Bibliography: `thesis/main.bib`.

The current corpus includes the Nature semantic entropy paper, the ICLR semantic uncertainty precursor, Semantic Entropy Probes, Kernel Language Entropy, SelfCheckGPT, P(True), hallucination/UQ surveys, dataset citations, model citations, and newer related work on semantic energy, semantic volume, high-certainty hallucinations, and neuron/probe-based detection.

## Data and Artifacts

Do not commit large data, raw datasets, model outputs, checkpoints, downloaded models, or bulk generated artifacts. Keep placeholders such as `data/raw/README.md` and `results/README.md` tracked so the intended layout remains visible.
