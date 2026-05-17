"""Sampling harness for Phase 1 pipeline plumbing.

This module bridges the dataset ingestion layer (:mod:`semantic_entropy.datasets`)
and the model sampling / scoring steps. It defines :class:`RunConfig`, which
bundles every piece of run metadata needed to produce a fully-specified JSONL
record, and a chain of pipeline functions:

* :func:`make_presampling_record` — converts a
  :class:`~semantic_entropy.datasets.PromptItem` + :class:`RunConfig` into a
  *pre-sampling* dict with all schema keys present and pending fields left
  empty or ``None``.
* :func:`sample_record` — calls a caller-supplied model function to fill
  ``sampled_answers``, then derives ``normalized_answers``,
  ``semantic_clusters``, and ``cluster_representatives`` using the
  current clustering backend.
* :func:`evaluate_correctness` — sets ``correctness_label`` (bool) by
  checking whether any normalized sampled answer matches any normalized
  reference answer.
* :func:`run_pipeline` — chains all of the above with entropy scoring and
  schema validation, writing fully-scored JSONL records to disk.

Pre-sampling records produced by :func:`make_presampling_record` are
intentionally *not* valid against
:func:`semantic_entropy.schema.validate_record` — they are an intermediate
format used only within the pipeline. The validation gate applies only to
records that have passed every downstream step, as enforced inside
:func:`run_pipeline`.

The module also provides :func:`make_presampling_jsonl` for writing
pre-sampling records without running inference, and a :func:`main` CLI
entry point registered as ``semantic-entropy-make-presampling-jsonl`` in
``pyproject.toml``.

**Example pipeline usage**::

    from semantic_entropy.harness import RunConfig, run_pipeline
    from semantic_entropy.datasets import load_triviaqa_records

    def my_model(prompt: str, num_samples: int) -> list[str]:
        # Replace with real model call
        return [f"Paris (sample {i})" for i in range(num_samples)]

    config = RunConfig(run_id="pilot-001", model="my-model", model_tier="small")
    items = load_triviaqa_records()
    n = run_pipeline(items, config, my_model, Path("out/scored.jsonl"))
    print(f"wrote {n} records")

**Example CLI usage**::

    # Both stubs, default smoke config
    semantic-entropy-make-presampling-jsonl out/presampling.jsonl

    # TriviaQA only, custom run ID
    semantic-entropy-make-presampling-jsonl --dataset triviaqa \\
        --run-id pilot-001 out/tqa_presampling.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from semantic_entropy.cluster_check import check_cluster_consistency
from semantic_entropy.clustering import exact_match_cluster
from semantic_entropy.datasets import (
    DatasetError,
    PromptItem,
    SVAMP_STUB_PATH,
    TRIVIAQA_STUB_PATH,
    load_svamp_records,
    load_triviaqa_records,
)
from semantic_entropy.normalization import normalize_answers
from semantic_entropy.schema import SchemaError, validate_record
from semantic_entropy.scoring import score_record


# ---------------------------------------------------------------------------
# RunConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    """Immutable bundle of run-level metadata for a Phase 1/2 experiment.

    All fields except *run_id* have sensible defaults matching the Phase 1
    smoke configuration described in ``docs/experiment_protocol.md``.

    Attributes:
        run_id: Unique identifier for this run, e.g. ``"smoke-001"`` or
            ``"phase1-tinygpu-7b-seed0"``.
        phase: One of ``"phase1"`` or ``"phase2"``.
        cluster: Compute cluster name: ``"local"``, ``"tinygpu"``,
            ``"alex"``, or ``"helma"``.
        model: Model name or path string, e.g. ``"synthetic-model"`` or
            ``"meta-llama/Llama-3.1-8B-Instruct"``.
        model_tier: One of ``"small"``, ``"7b_8b"``, or ``"70b_plus"``.
        entailment_backend: NLI backend used for semantic clustering, e.g.
            ``"synthetic-preclustered"`` or ``"cross-encoder/nli-deberta-v3-base"``.
        num_samples: Number of model samples per prompt (``M`` in the thesis).
        temperature: Decoding temperature.
        top_p: Nucleus sampling threshold.
        max_new_tokens: Maximum number of new tokens per sample.
        seed: Global RNG seed for reproducibility.
    """

    run_id: str
    phase: str = "phase1"
    cluster: str = "local"
    model: str = "synthetic-model"
    model_tier: str = "small"
    entailment_backend: str = "synthetic-preclustered"
    num_samples: int = 4
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 64
    seed: int = 0


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def make_presampling_record(item: PromptItem, config: RunConfig) -> dict[str, Any]:
    """Return a pre-sampling record dict for *item* under *config*.

    The returned dict has every top-level field defined in the experimental
    JSONL schema, with the following pending fields left empty or ``None``
    to be filled by downstream pipeline steps:

    * ``sampled_answers`` — populated by the model sampling step.
    * ``normalized_answers`` — populated by the normalization step.
    * ``semantic_clusters`` — populated by the NLI clustering step.
    * ``cluster_representatives`` — populated by the NLI clustering step.
    * ``scores`` — populated by the scoring step.
    * ``correctness_label`` — populated after answer evaluation.

    Args:
        item: Source prompt extracted from a dataset.
        config: Run-level metadata and decoding hyperparameters.

    Returns:
        A ``dict`` with all schema-level keys present. The ``scores``
        sub-dict is an empty mapping and ``correctness_label`` is ``None``.
    """
    return {
        "run_id": config.run_id,
        "phase": config.phase,
        "cluster": config.cluster,
        "prompt_id": item.prompt_id,
        "dataset": item.dataset,
        "split": item.split,
        "prompt": item.prompt,
        "reference_answers": list(item.reference_answers),
        "model": config.model,
        "model_tier": config.model_tier,
        "decoding": {
            "num_samples": config.num_samples,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_new_tokens": config.max_new_tokens,
            "seed": config.seed,
        },
        "sampled_answers": [],
        "normalized_answers": [],
        "semantic_clusters": [],
        "cluster_representatives": [],
        "scores": {},
        "entailment_backend": config.entailment_backend,
        "correctness_label": None,
    }


def make_presampling_jsonl(
    items: Iterable[PromptItem],
    config: RunConfig,
    output_path: Path,
) -> int:
    """Write pre-sampling records to *output_path* in JSONL format.

    Args:
        items: Sequence of :class:`~semantic_entropy.datasets.PromptItem`
            objects to convert.
        config: Run-level metadata applied to every record.
        output_path: Destination file path. Created or overwritten.

    Returns:
        The number of records written.

    Raises:
        OSError: If *output_path* cannot be opened for writing.
    """
    records_written = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for item in items:
            record = make_presampling_record(item, config)
            fh.write(json.dumps(record, sort_keys=True) + "\n")
            records_written += 1
    return records_written


# ---------------------------------------------------------------------------
# Sampling, correctness evaluation, and end-to-end pipeline
# ---------------------------------------------------------------------------

#: Type alias for a model sampling function.
#:
#: A ``ModelFn`` receives a *prompt* string and the requested number of
#: independent samples *num_samples*, and returns a list of exactly
#: *num_samples* answer strings. Each string is one raw model generation
#: (untokenized, untruncated beyond the caller's ``max_new_tokens`` budget).
#:
#: For smoke testing, a trivial implementation suffices::
#:
#:     def stub_model(prompt: str, num_samples: int) -> list[str]:
#:         return ["Paris"] * num_samples
ModelFn = Callable[[str, int], list[str]]


def sample_record(record: dict[str, Any], model_fn: ModelFn) -> dict[str, Any]:
    """Fill the sampling fields of a pre-sampling record.

    Calls *model_fn* with the record's prompt and requested sample count,
    normalizes the returned answers, and assigns cluster IDs using the
    current clustering backend (:func:`semantic_entropy.clustering.exact_match_cluster`).
    The input record is not modified; a new dict is returned.

    The fields populated are:

    * ``sampled_answers`` — raw strings returned by *model_fn*.
    * ``normalized_answers`` — canonicalized forms via
      :func:`semantic_entropy.normalization.normalize_answers`.
    * ``semantic_clusters`` — integer cluster IDs, one per sample.
    * ``cluster_representatives`` — one representative string per cluster.

    The fields ``scores``, ``correctness_label``, and all run-metadata
    fields are passed through unchanged.

    Args:
        record: A pre-sampling record dict as produced by
            :func:`make_presampling_record`.  Must contain ``"prompt"`` and
            ``"decoding"`` with a positive ``"num_samples"`` key.
        model_fn: Callable ``(prompt, num_samples) -> list[str]``.  Must
            return a list whose length equals *num_samples*; a
            :class:`ValueError` is raised otherwise.

    Returns:
        A shallow copy of *record* with the four sampling fields populated.

    Raises:
        ValueError: If *model_fn* returns the wrong number of answers.
        KeyError: If *record* is missing ``"prompt"`` or
            ``"decoding.num_samples"``.
    """
    prompt: str = record["prompt"]
    num_samples: int = record["decoding"]["num_samples"]

    sampled = model_fn(prompt, num_samples)
    if len(sampled) != num_samples:
        raise ValueError(
            f"model_fn returned {len(sampled)} answers; expected {num_samples}"
        )

    normalized = normalize_answers(sampled)
    cluster_ids, representatives = exact_match_cluster(normalized)

    return {
        **record,
        "sampled_answers": list(sampled),
        "normalized_answers": normalized,
        "semantic_clusters": cluster_ids,
        "cluster_representatives": representatives,
    }


def evaluate_correctness(record: dict[str, Any]) -> dict[str, Any]:
    """Set ``correctness_label`` based on normalized answer/reference overlap.

    A record is considered *correct* if at least one normalized sampled
    answer is an exact string match for at least one normalized reference
    answer. Both sets of answers are normalized with
    :func:`semantic_entropy.normalization.normalize_answers` before
    comparison, so surface differences in casing and whitespace are ignored.

    This is the surface-form exact-match correctness criterion used by the
    Phase 1 evaluation and is appropriate for both TriviaQA (which accepts
    multiple aliases) and SVAMP (which has a single numeric reference). It
    will under-score cases where a semantically correct paraphrase does not
    normalize to the reference form; that limitation is addressed in
    Stage 3 by the NLI-based clustering and a richer normalizer.

    Args:
        record: A record dict that already has ``normalized_answers`` and
            ``reference_answers`` populated.

    Returns:
        A shallow copy of *record* with ``correctness_label`` set to a
        ``bool``.
    """
    normalized_refs = set(normalize_answers(record["reference_answers"]))
    correct = any(ans in normalized_refs for ans in record["normalized_answers"])
    return {**record, "correctness_label": correct}


def run_pipeline(
    items: Iterable[PromptItem],
    config: RunConfig,
    model_fn: ModelFn,
    output_path: Path,
) -> int:
    """Run the full Phase 1 pipeline and write schema-valid scored JSONL.

    For each :class:`~semantic_entropy.datasets.PromptItem` in *items* the
    pipeline executes in order:

    1. :func:`make_presampling_record` — populate run metadata and prompt fields.
    2. :func:`sample_record` — call *model_fn*, normalize, cluster.
    3. :func:`evaluate_correctness` — set ``correctness_label``.
    4. :func:`semantic_entropy.scoring.score_record` — compute entropy scores.
    5. :func:`semantic_entropy.schema.validate_record` and
       :func:`semantic_entropy.cluster_check.check_cluster_consistency` —
       validate the fully-scored record.
    6. Write one JSON line (keys sorted) to *output_path*.

    Args:
        items: Source prompts to process.
        config: Run-level metadata applied to every record.
        model_fn: Model sampling function; see :data:`ModelFn`.
        output_path: Destination JSONL file. Created or overwritten.

    Returns:
        The number of records successfully written.

    Raises:
        ValueError: If *model_fn* returns the wrong number of answers for
            any prompt.
        SchemaError: If a fully-scored record fails schema or cluster
            consistency validation, indicating a pipeline bug.
        OSError: If *output_path* cannot be opened for writing.
    """
    records_written = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for item in items:
            record: dict[str, Any] = make_presampling_record(item, config)
            record = sample_record(record, model_fn)
            record = evaluate_correctness(record)
            record["scores"] = score_record(
                record["sampled_answers"],
                record["normalized_answers"],
                record["semantic_clusters"],
            )
            try:
                validate_record(record)
                check_cluster_consistency(record)
            except SchemaError as exc:
                raise SchemaError(
                    f"pipeline validation failed for prompt_id={record.get('prompt_id')!r}: {exc}"
                ) from exc
            fh.write(json.dumps(record, sort_keys=True) + "\n")
            records_written += 1
    return records_written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_DATASET_CHOICES = ("triviaqa", "svamp", "all")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: write pre-sampling JSONL from dataset stubs.

    Reads the committed stub files for the requested dataset(s) and writes
    one pre-sampling record per prompt to *output_jsonl*. All run-metadata
    fields can be overridden via command-line flags; sensible Phase 1 smoke
    defaults apply when flags are omitted.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate a pre-sampling JSONL file from TriviaQA and/or SVAMP "
            "stub data. The output records have all schema fields populated "
            "except sampled_answers, normalized_answers, semantic_clusters, "
            "cluster_representatives, scores, and correctness_label, which "
            "are left empty or null for the downstream sampling harness to fill."
        )
    )
    parser.add_argument("output_jsonl", type=Path, metavar="output.jsonl")
    parser.add_argument(
        "--dataset",
        choices=_DATASET_CHOICES,
        default="all",
        help="Which stub(s) to load (default: all).",
    )
    parser.add_argument(
        "--run-id",
        default="smoke-presampling-001",
        help="Run identifier written into every record (default: smoke-presampling-001).",
    )
    parser.add_argument("--phase", default="phase1", choices=("phase1", "phase2"))
    parser.add_argument(
        "--cluster",
        default="local",
        choices=("local", "tinygpu", "alex", "helma"),
    )
    parser.add_argument("--model", default="synthetic-model")
    parser.add_argument(
        "--model-tier",
        default="small",
        choices=("small", "7b_8b", "70b_plus"),
        dest="model_tier",
    )
    parser.add_argument(
        "--entailment-backend",
        default="synthetic-preclustered",
        dest="entailment_backend",
    )
    parser.add_argument("--num-samples", type=int, default=4, dest="num_samples")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95, dest="top_p")
    parser.add_argument(
        "--max-new-tokens", type=int, default=64, dest="max_new_tokens"
    )
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(argv)

    config = RunConfig(
        run_id=args.run_id,
        phase=args.phase,
        cluster=args.cluster,
        model=args.model,
        model_tier=args.model_tier,
        entailment_backend=args.entailment_backend,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    try:
        items = list(_load_items(args.dataset))
    except DatasetError as error:
        parser.exit(status=1, message=f"error loading dataset: {error}\n")

    try:
        n = make_presampling_jsonl(items, config, args.output_jsonl)
    except OSError as error:
        parser.exit(status=1, message=f"error writing output: {error}\n")

    print(f"wrote {n} pre-sampling records to {args.output_jsonl}")
    return 0


def _load_items(dataset: str) -> Iterable[PromptItem]:
    if dataset in ("triviaqa", "all"):
        yield from load_triviaqa_records(TRIVIAQA_STUB_PATH)
    if dataset in ("svamp", "all"):
        yield from load_svamp_records(SVAMP_STUB_PATH)


if __name__ == "__main__":
    raise SystemExit(main())
