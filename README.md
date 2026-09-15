# Detecting Hallucinations in Large Language Models Using Semantic Entropy

Bachelor's thesis by Jakub Wiśniewski, Mathematical Institute for Machine
Learning and Data Science, Catholic University of Eichstätt-Ingolstadt, 2026.

The thesis reimplements semantic entropy (Farquhar et al., Nature 2024,
[doi:10.1038/s41586-024-07421-0](https://doi.org/10.1038/s41586-024-07421-0))
and compares it with simpler uncertainty scores on TriviaQA, NQ-Open, SVAMP and
a biography task. The answers come from Mistral-7B-Instruct-v0.3,
Llama-3.1-8B-Instruct and Llama-3.1-70B-Instruct.

This repository holds the LaTeX source, the code that ran the experiments, and
the measurement files that all tables and figures are built from.

## Contents

| Path | What it holds |
|---|---|
| `thesis/` | LaTeX source, figures and generated tables |
| `data/` | Measurement files the tables are built from, see `data/README.md` |
| `results/` | Paired bootstrap intervals and the outputs of the checks in Appendix C |
| `src/semantic_entropy/` | Python package: sampling, clustering, entropy scores, P(True), probes, metrics |
| `scripts/` | Data preparation, job submission, export, analysis, and the table and figure builders |
| `slurm/` | Batch scripts used on the NHR@FAU Alex and Fritz clusters |
| `tests/` | Tests for the package and the scripts |
| `external/factscore/topics.json` | Subjects of the biography prompts |

Raw generated answers, hidden states and model weights are not included.

## Rebuild the tables, figures and PDF

This needs Python 3.11 or newer with [uv](https://docs.astral.sh/uv/), and TeX
Live with `latexmk` and Biber. No GPU or model weights are needed.

```sh
uv sync --locked --extra test --extra plot --extra sep
uv run python scripts/build_thesis_tables.py data/measurements.csv thesis/includes
uv run python scripts/build_robustness_tables.py
uv run python scripts/plot_method_schematic.py
cd thesis && latexmk -pdf main.tex
```

To check the files:

```sh
uv run python scripts/check_thesis_measurements.py     # complete grid, intervals match the measurements
uv run python scripts/check_figure_table_freshness.py  # tables and figures match a fresh rebuild (needs pdftoppm)
uv run pytest
```

## How the experiments were run

The experiments ran with Slurm on the NHR@FAU clusters. The launchers contain
settings of those clusters (paths, model cache, accounts) and need changes to
run elsewhere.

Short-answer study:

1. `scripts/prepare_replication_data.py` loads the datasets and selects the
   questions the way the released code does.
2. `scripts/submit_replication_grid.sh` generates the answers for the two
   original prompting conditions, `scripts/submit_factorial.sh` for the two
   added ones, and `scripts/submit_temp_sweep.sh` for the lower temperatures.
3. `scripts/submit_ladder.sh` and `scripts/submit_ladder_sweep.sh` cluster the
   stored answers with each entailment model.
4. The `scripts/submit_relabel*.sh` scripts grade the answers with the LLM
   judges.
5. `scripts/submit_ptrue.sh` computes P(True).
6. `scripts/submit_probes.sh` trains the probes and `scripts/apply_probes.sh`
   scores the evaluation questions.
7. `scripts/export_data_package_v2.py` writes the files in `data/`, and
   `scripts/paired_bootstrap_hierarchical.py` computes the paired intervals.

Biography study:

1. `scripts/prepare_bio.py` builds the prompts and the Wikipedia references.
2. `slurm/bio_v3_alex.sbatch` generates the answers, and
   `scripts/dispatch_bio_v3_phase4.sh` submits the clustering and grading jobs.
3. `scripts/join_and_summarize_bio_v3.py` joins the labels, and
   `scripts/paired_bootstrap_bio_records.py` computes the intervals.

Appendix C checks: `scripts/export_robustness_inputs.py`,
`scripts/analyze_robustness.py`, `scripts/cache_robustness_pairs.py` and
`scripts/analyze_clustering_robustness.py`. The clustering checks can be rerun
from the files in `results/robustness/`:

```sh
uv run python scripts/analyze_clustering_robustness.py \
  results/robustness/clustering_inputs.json.gz \
  results/robustness/pair_cache.json.gz /tmp/clustering-check
```
