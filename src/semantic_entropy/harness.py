"""Pre-sampling harness for Phase 1 pipeline plumbing.

This module bridges the dataset ingestion layer (:mod:`semantic_entropy.datasets`)
and the model sampling step. It defines :class:`RunConfig`, which bundles every
piece of run metadata needed to produce a fully-specified JSONL record, and
:func:`make_presampling_record`, which takes a :class:`~semantic_entropy.datasets.PromptItem`
and a :class:`RunConfig` and returns a *pre-sampling* record dict — a record
that has all structural fields populated but whose ``sampled_answers``,
``normalized_answers``, ``semantic_clusters``, ``cluster_representatives``,
``scores``, and ``correctness_label`` fields are empty or ``null``, awaiting
the outputs of the model sampling, normalization, NLI clustering, scoring, and
correctness-evaluation steps respectively.

Pre-sampling records are intentionally *not* valid against
:func:`semantic_entropy.schema.validate_record` — they are an intermediate
format used only within the pipeline. The validation gate applies only to
fully-scored records that have passed all downstream steps.

The module also provides :func:`make_presampling_jsonl` for writing a sequence
of :class:`~semantic_entropy.datasets.PromptItem` objects to a JSONL file, and
a :func:`main` CLI entry point registered as
``semantic-entropy-make-presampling-jsonl`` in ``pyproject.toml``.

**Example CLI usage**::

    # Both stubs, default smoke config
    semantic-entropy-make-presampling-jsonl out/presampling.jsonl

    # TriviaQA only, custom run ID
    semantic-entropy-make-presampling-jsonl --dataset triviaqa \\
        --run-id pilot-001 out/tqa_presampling.jsonl

    # SVAMP only, 10 samples, seed 42
    semantic-entropy-make-presampling-jsonl --dataset svamp \\
        --num-samples 10 --seed 42 out/svamp_presampling.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from semantic_entropy.datasets import (
    DatasetError,
    PromptItem,
    SVAMP_STUB_PATH,
    TRIVIAQA_STUB_PATH,
    load_svamp_records,
    load_triviaqa_records,
)


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
