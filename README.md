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

The scientific-correction v2 execution pipeline is implemented as of
2026-07-17, but the cluster runs have not yet been executed. The current
PDF and historical result tree are not submission-ready: the paired table mixed
incompatible summaries, the reported SEP artifacts were correctness-trained
accuracy probes, and the biography labels were automated proxies rather than
the published FactualBio human labels. Historical outputs are retained for
audit but are not authoritative primary evidence.

The correction keeps the fixed seeds 42, 43, and 44. The fresh SEP-v2
evaluation collections also supply the primary short-answer generations after
their complete collection-to-reclustering lineage passes the v2 protocol lock.
The correction reclusters them with question-conditioned DeBERTa-v3-large,
reruns a genuine entropy-target, fixed-final-layer SEP adaptation on disjoint
train/evaluation records, and uses the 150 manually labeled claims in the
published FactualBio release in a fixed-claim, question-conditioned adaptation.
Neither extension is presented as an exact paper replication. See
`PLANS.md` for the locked scientific scope. The canonical execution paths are:

- `scripts/lock_primary_shortform_v2.py` and
  `scripts/export_primary_shortform_v2.py` for the fail-closed 27-cell
  short-answer family;
- `scripts/dispatch_seeded_sep.sh` for genuine SEP collection, fitting, and
  held-out scoring; and
- `scripts/run_published_factualbio_v2.sh` for the published-label FactualBio
  extension.

The six complete SEP source JSONLs are intentionally not present in this local
checkout. Authenticate to Alex and synchronize the authoritative processed
files before creating a lock:

```sh
ssh alex
rsync -av alex:~/thesis/data/processed/ data/processed/
```

The manifest independently requires exactly 500/500 TriviaQA, 500/500 NQ-Open,
and 700/300 SVAMP train/evaluation records, so a partial synchronization cannot
become authoritative. Export the exact 40-character revisions from the cached
model snapshots before either a command preview or a real submission:

```sh
export CODE_COMMIT="$(git rev-parse HEAD)"
export MISTRAL7B_MODEL_REVISION=<40-hex-sha>
export MISTRAL7B_TOKENIZER_REVISION=<40-hex-sha>
export LLAMA8B_MODEL_REVISION=<40-hex-sha>
export LLAMA8B_TOKENIZER_REVISION=<40-hex-sha>
export LLAMA70B_MODEL_REVISION=<40-hex-sha>
export LLAMA70B_TOKENIZER_REVISION=<40-hex-sha>
export NLI_MODEL_REVISION=<40-hex-sha>
export NLI_TOKENIZER_REVISION="$NLI_MODEL_REVISION"
```

Create the immutable protocol manifest from the complete sources and those
revisions:

```sh
PYTHONPATH=src python3 scripts/sep_v2_manifest.py create \
  --output results_sep_v2_protocol/sep_v2_protocol_manifest.json \
  --code-commit "$CODE_COMMIT" \
  --mistral7b-model-revision "$MISTRAL7B_MODEL_REVISION" \
  --mistral7b-tokenizer-revision "$MISTRAL7B_TOKENIZER_REVISION" \
  --llama8b-model-revision "$LLAMA8B_MODEL_REVISION" \
  --llama8b-tokenizer-revision "$LLAMA8B_TOKENIZER_REVISION" \
  --llama70b-model-revision "$LLAMA70B_MODEL_REVISION" \
  --llama70b-tokenizer-revision "$LLAMA70B_TOKENIZER_REVISION" \
  --nli-model-revision "$NLI_MODEL_REVISION" \
  --nli-tokenizer-revision "$NLI_TOKENIZER_REVISION" \
  --triviaqa-max-new-tokens "${TRIVIAQA_SEP_MAX_NEW_TOKENS:-64}" \
  --nqopen-max-new-tokens "${NQOPEN_SEP_MAX_NEW_TOKENS:-64}" \
  --svamp-max-new-tokens "${SVAMP_SEP_MAX_NEW_TOKENS:-32}" \
  --feature-response-max-new-tokens 256
```

The 64/64/32 values are ceilings for the ten stochastic responses used to
calculate semantic entropy. The separately generated greedy SEP feature
response has a fixed 256-token ceiling and must still terminate with EOS/EOT;
reaching 256 fails the collection cell. The July 18 pilot established that a
shared 64-token ceiling was insufficient for this response, so manifests from
the earlier commit and partial pilot roots are diagnostic artifacts only and
must not be reused.

The command refuses to overwrite an existing manifest. Set
`SEP_V2_MANIFEST=results_sep_v2_protocol/sep_v2_protocol_manifest.json` for any
`DRY_RUN=0` dispatch; the dispatcher rehashes every source and rejects revision
or configuration drift before calling `sbatch`. Collection jobs recheck their
source hash after leaving the queue, and collection, reclustering, and probe
sidecars retain the verified manifest hash. `DRY_RUN=1` is only a data-free
command preview; it still requires the revision variables above.

The pilot and full matrix must use different, initially empty result roots.
Sharing a root would correctly fail the exact-grid completion seal:

```sh
export SEP_V2_MANIFEST=results_sep_v2_protocol/sep_v2_protocol_manifest.json
SE_RESULTS_ROOT=results_sep_v2_pilot PILOT=1 DRY_RUN=1 \
  bash scripts/dispatch_seeded_sep.sh
SE_RESULTS_ROOT=results_sep_v2_pilot PILOT=1 DRY_RUN=0 \
  bash scripts/dispatch_seeded_sep.sh

PYTHONPATH=src python3 scripts/sep_v2_manifest.py pilot-complete \
  --protocol-manifest "$SEP_V2_MANIFEST" \
  --results-root results_sep_v2_pilot \
  --output results_sep_v2_pilot/pilot_artifact_manifest.json

SE_RESULTS_ROOT=results_sep_v2_full PILOT=0 DRY_RUN=1 \
  bash scripts/dispatch_seeded_sep.sh
SE_RESULTS_ROOT=results_sep_v2_full PILOT=0 DRY_RUN=0 \
  bash scripts/dispatch_seeded_sep.sh
```

After all full jobs finish, aggregate and seal the fresh 54-run probe grid:

```sh
PYTHONPATH=src python3 scripts/aggregate_results.py \
  --results-dir results_sep_v2_full \
  --output-csv results_sep_v2_full/tables/metric_summary.csv \
  --output-json results_sep_v2_full/tables/metric_summary.json
PYTHONPATH=src python3 scripts/summarize_multiseed_results.py \
  --summary-csv results_sep_v2_full/tables/metric_summary.csv \
  --tables-dir results_sep_v2_full/tables \
  --includes-dir /tmp/sep_v2_includes
PYTHONPATH=src python3 scripts/sep_v2_manifest.py complete \
  --protocol-manifest "$SEP_V2_MANIFEST" \
  --results-root results_sep_v2_full \
  --summary-csv results_sep_v2_full/tables/multiseed_method_summary.csv \
  --output results_sep_v2_full/tables/sep_v2_artifact_manifest.json
```

The same 27 fresh evaluation collection/reclustering pairs are the only
permitted source for the primary DSE-versus-surface analysis. Create an
explicit `primary_shortform_v2_selection.json`; no script searches for a
"latest" run. The selection has this shape, with exactly one cell for every
dataset/model/seed combination:

```json
{
  "schema_version": "primary-shortform-v2-selection-v1",
  "locked_at": "2026-07-17T18:00:00+02:00",
  "analysis_code_commit": "<CODE_COMMIT>",
  "derived_qcond_code_commit": "<CODE_COMMIT>",
  "sep_v2_protocol_manifest": "results_sep_v2_protocol/sep_v2_protocol_manifest.json",
  "sep_v2_protocol_manifest_sha256": "<SHA-256>",
  "dataset_specs": {
    "triviaqa": {"split": "<split>", "artifact_path": "<eval JSONL>"},
    "svamp": {"split": "<split>", "artifact_path": "<eval JSONL>"},
    "nqopen": {"split": "<split>", "artifact_path": "<eval JSONL>"}
  },
  "model_specs": {
    "mistralai/Mistral-7B-Instruct-v0.3": {
      "model_revision": "<40-hex-sha>",
      "tokenizer_id": "mistralai/Mistral-7B-Instruct-v0.3",
      "tokenizer_revision": "<40-hex-sha>"
    },
    "meta-llama/Llama-3.1-8B-Instruct": {"model_revision": "<40-hex-sha>", "tokenizer_id": "meta-llama/Llama-3.1-8B-Instruct", "tokenizer_revision": "<40-hex-sha>"},
    "meta-llama/Llama-3.1-70B-Instruct": {"model_revision": "<40-hex-sha>", "tokenizer_id": "meta-llama/Llama-3.1-70B-Instruct", "tokenizer_revision": "<40-hex-sha>"}
  },
  "nli_spec": {
    "model_id": "cross-encoder/nli-deberta-v3-large",
    "model_revision": "<40-hex-sha>",
    "tokenizer_id": "cross-encoder/nli-deberta-v3-large",
    "tokenizer_revision": "<40-hex-sha>",
    "question_conditioned": true,
    "clustering_rule": "bidirectional"
  },
  "cells": [
    {
      "dataset": "triviaqa",
      "model": "mistralai/Mistral-7B-Instruct-v0.3",
      "seed": 42,
      "source": {"scored_path": "<eval collection scored.jsonl>", "job_meta_path": "<eval collection job_meta.json>"},
      "derived": {"scored_path": "<eval q-conditioned scored.jsonl>", "job_meta_path": "<eval q-conditioned job_meta.json>"}
    }
  ]
}
```

The displayed `cells` array is abbreviated; the lock rejects anything other
than the exact 27-cell grid. Every path must be repository-relative. The lock
rehashes the protocol and all artifacts, invokes the SEP-v2 verifier on all 54
collection/reclustering bindings, joins every source record back to the frozen
dataset, recomputes both entropies and AUROCs, and refuses existing outputs:

```sh
mkdir -p results_authoritative_v2/tables
PYTHONPATH=src python3 scripts/lock_primary_shortform_v2.py \
  primary_shortform_v2_selection.json \
  --repo-root . \
  --normalized-sidecars-dir results_authoritative_v2/tables/primary_sidecars \
  --output-manifest results_authoritative_v2/tables/primary_shortform_v2_manifest.json \
  --output-family-csv results_authoritative_v2/tables/primary_shortform_v2_family.csv

PYTHONPATH=src python3 scripts/fixed_seed_auroc_bootstrap.py \
  results_authoritative_v2/tables/primary_shortform_v2_family.csv \
  --source-manifest results_authoritative_v2/tables/primary_shortform_v2_manifest.json \
  --expected-comparison-id "triviaqa::mistralai/Mistral-7B-Instruct-v0.3::question-conditioned-dse-vs-surface" \
  --expected-comparison-id "triviaqa::meta-llama/Llama-3.1-8B-Instruct::question-conditioned-dse-vs-surface" \
  --expected-comparison-id "triviaqa::meta-llama/Llama-3.1-70B-Instruct::question-conditioned-dse-vs-surface" \
  --expected-comparison-id "svamp::mistralai/Mistral-7B-Instruct-v0.3::question-conditioned-dse-vs-surface" \
  --expected-comparison-id "svamp::meta-llama/Llama-3.1-8B-Instruct::question-conditioned-dse-vs-surface" \
  --expected-comparison-id "svamp::meta-llama/Llama-3.1-70B-Instruct::question-conditioned-dse-vs-surface" \
  --expected-comparison-id "nqopen::mistralai/Mistral-7B-Instruct-v0.3::question-conditioned-dse-vs-surface" \
  --expected-comparison-id "nqopen::meta-llama/Llama-3.1-8B-Instruct::question-conditioned-dse-vs-surface" \
  --expected-comparison-id "nqopen::meta-llama/Llama-3.1-70B-Instruct::question-conditioned-dse-vs-surface" \
  --resamples 10000 \
  --output results_authoritative_v2/tables/primary_shortform_v2_fixed_seed_stats.json
```

Do not use `scripts/plan_full_rerun_jobs.py` for the correction's SEP or
biography stages; that planner is retained only to audit the historical
workflow.

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

## Published FactualBio Labels

The primary long-form correction pins release `v.1.0.0` of
`jlko/long_hallucinations` at commit
`957b806033d0b769db34beb7ce34316bd498f373`. Importing the source does not
execute the upstream Python file:

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

The importer requires exactly 21 biographies, 150 claim annotations, and the
published 105 correct / 45 false label split before writing output.

The actual published-label run uses one new output root and 19 blocking Slurm
jobs. The helper question model is explicit rather than silently inferred:

```sh
export FACTUALBIO_QUESTION_MODEL=Qwen/Qwen2.5-72B-Instruct
export FACTUALBIO_QUESTION_REVISION=<40-hex-sha>
bash scripts/run_published_factualbio_v2.sh \
  --source-py data/raw/factualbio_data_v1.0.0.py \
  --output-root results_factualbio_v2_full \
  --code-commit "$CODE_COMMIT" \
  --question-model "$FACTUALBIO_QUESTION_MODEL" \
  --question-revision "$FACTUALBIO_QUESTION_REVISION" \
  --mistral-revision "$MISTRAL7B_MODEL_REVISION" \
  --llama8b-revision "$LLAMA8B_MODEL_REVISION" \
  --llama70b-revision "$LLAMA70B_MODEL_REVISION" \
  --nli-revision "$NLI_MODEL_REVISION" \
  --dry-run
# Remove --dry-run only after checking the 19 printed submissions.
```

The command preview performs no model or artifact validation. A real run
refuses an existing output root, executes the import and every validation gate,
and writes `aggregate/factualbio_summary.csv` plus `artifact_manifest.json`.

After the SEP, primary, and FactualBio completion seals exist, generate the
authoritative tables in one pass:

```sh
PYTHONPATH=src python3 scripts/build_thesis_tables_from_summary.py \
  --tables-dir results_authoritative_v2/tables \
  --primary-family-csv results_authoritative_v2/tables/primary_shortform_v2_family.csv \
  --primary-stats-json results_authoritative_v2/tables/primary_shortform_v2_fixed_seed_stats.json \
  --primary-manifest results_authoritative_v2/tables/primary_shortform_v2_manifest.json \
  --sep-v2-manifest "$SEP_V2_MANIFEST" \
  --sep-v2-summary results_sep_v2_full/tables/multiseed_method_summary.csv \
  --sep-v2-completion-manifest results_sep_v2_full/tables/sep_v2_artifact_manifest.json \
  --factualbio-v2-summary results_factualbio_v2_full/aggregate/factualbio_summary.csv \
  --factualbio-v2-manifest results_factualbio_v2_full/artifact_manifest.json \
  --out-csv-dir results_authoritative_v2/tables/thesis_formatted \
  --out-tex-dir thesis/includes
```

This authoritative mode excludes all historical non-probe summary rows. The
old result tree remains available only through the explicitly named legacy
mode; it cannot enter the new rankings by accident. Table generation is not a
substitute for revising the affected prose and figures before rebuilding and
visually checking `thesis/main.pdf`.

## Build the Thesis

```sh
cd thesis && latexmk -pdf main.tex
```

## Run the Tests

```sh
uv run pytest tests/ -q
```
