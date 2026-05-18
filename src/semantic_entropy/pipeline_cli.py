"""CLI entry point for the full semantic-entropy sampling pipeline.

This module provides the ``semantic-entropy-run-pipeline`` command, which
wraps :func:`~semantic_entropy.harness.run_pipeline` with dataset loading,
model adapter resolution, and NLI entailment-function hooks.

**Smoke testing** with the synthetic stub model::

    semantic-entropy-run-pipeline --dataset triviaqa --stub-model \\
        /tmp/smoke_scored.jsonl

**Running with a user-supplied model module** (Stage 5)::

    semantic-entropy-run-pipeline --dataset triviaqa \\
        --model-module my_adapters:make_model \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --entailment-module my_adapters:make_nli \\
        results/phase1/tqa_scored.jsonl

The ``--model-module`` and ``--entailment-module`` flags accept a
``module_path:callable_name`` string.  The module is imported at runtime
and the named callable is invoked with no arguments, returning a
:data:`~semantic_entropy.models.ModelFn` or
:data:`~semantic_entropy.clustering.NliFn` respectively.

This design keeps the CLI itself stdlib-only; heavy ML dependencies live
entirely in the user-written adapter modules.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from semantic_entropy.clustering import NliFn
from semantic_entropy.datasets import (
    DatasetError,
    PromptItem,
    SVAMP_STUB_PATH,
    TRIVIAQA_STUB_PATH,
    load_svamp_records,
    load_triviaqa_records,
)
from semantic_entropy.harness import ModelFn, RunConfig, run_pipeline
from semantic_entropy.models import SyntheticModel
from semantic_entropy.schema import SchemaError


# ---------------------------------------------------------------------------
# Dynamic module loading
# ---------------------------------------------------------------------------


def load_callable(spec: str) -> Any:
    """Import *spec* as ``module_path:callable_name`` and return the callable.

    The *module_path* is a dotted Python module name (e.g.
    ``my_adapters.hf``) or a filesystem path ending in ``.py``.  When a
    filesystem path is given the parent directory is prepended to
    ``sys.path`` so the module can be found by the import machinery.

    Args:
        spec: A string in the format ``module_path:callable_name``.

    Returns:
        The resolved callable object.

    Raises:
        ValueError: If *spec* does not contain exactly one colon.
        ImportError: If the module cannot be imported.
        AttributeError: If the callable name is not found in the module.
    """
    if spec.count(":") != 1:
        raise ValueError(
            f"expected 'module_path:callable_name', got {spec!r}"
        )
    module_path, attr_name = spec.split(":")

    # Support direct filesystem paths (e.g. "adapters/hf.py:make_model")
    if module_path.endswith(".py"):
        fs_path = Path(module_path).resolve()
        if not fs_path.is_file():
            raise ImportError(f"module file not found: {fs_path}")
        parent = str(fs_path.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        module_path = fs_path.stem

    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


# ---------------------------------------------------------------------------
# Dataset loading helpers
# ---------------------------------------------------------------------------

_DATASET_CHOICES = ("triviaqa", "svamp", "all")


def _load_items(
    dataset: str,
    *,
    data_path: str | None = None,
    limit: int | None = None,
) -> list[PromptItem]:
    """Load prompt items from the requested dataset(s).

    Args:
        dataset: One of ``"triviaqa"``, ``"svamp"``, or ``"all"``.
        data_path: Optional path override for the dataset file.  When
            *dataset* is ``"all"`` this flag is ignored and both stubs
            are used.
        limit: If set, truncate the loaded items to at most *limit*.

    Returns:
        A list of :class:`PromptItem` objects.
    """
    items: list[PromptItem] = []

    if dataset in ("triviaqa", "all"):
        path = Path(data_path) if data_path and dataset != "all" else TRIVIAQA_STUB_PATH
        items.extend(load_triviaqa_records(path))

    if dataset in ("svamp", "all"):
        path = Path(data_path) if data_path and dataset != "all" else SVAMP_STUB_PATH
        items.extend(load_svamp_records(path))

    if limit is not None and limit > 0:
        items = items[:limit]

    return items


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------

# Default synthetic answer pools keyed by dataset name.
_SYNTHETIC_POOLS: dict[str, list[str]] = {
    "triviaqa": ["Paris", "paris", "Lyon"],
    "svamp": ["7", "seven", "8"],
}
_DEFAULT_POOL = ["answer_a", "answer_b", "answer_c"]


def _make_stub_model(seed: int = 0) -> ModelFn:
    """Return a :class:`SyntheticModel` for smoke testing."""
    return SyntheticModel(_DEFAULT_POOL, seed=seed)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for ``semantic-entropy-run-pipeline``."""

    parser = argparse.ArgumentParser(
        prog="semantic-entropy-run-pipeline",
        description=(
            "Run the full semantic-entropy sampling pipeline: load a "
            "dataset, sample from a model (or synthetic stub), cluster, "
            "evaluate correctness, score, validate, and write JSONL."
        ),
    )

    # -- Positional output path ------------------------------------------------
    parser.add_argument(
        "output_jsonl",
        type=Path,
        metavar="output.jsonl",
        help="Destination path for scored JSONL output.",
    )

    # -- Dataset ---------------------------------------------------------------
    parser.add_argument(
        "--dataset",
        choices=_DATASET_CHOICES,
        default="all",
        help="Which dataset(s) to process (default: all).",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        dest="data_path",
        help=(
            "Path to a dataset JSONL file. Overrides the default stub path "
            "for the chosen --dataset (ignored when --dataset=all)."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N prompt items (useful for quick smoke tests).",
    )

    # -- Model adapter ---------------------------------------------------------
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--stub-model",
        action="store_true",
        default=False,
        dest="stub_model",
        help=(
            "Use the built-in SyntheticModel stub instead of a real model. "
            "Useful for smoke testing the pipeline without torch."
        ),
    )
    model_group.add_argument(
        "--model-module",
        default=None,
        dest="model_module",
        metavar="MODULE:CALLABLE",
        help=(
            "Import a model factory as 'module_path:callable_name'. The "
            "callable is invoked with no arguments and must return a "
            "ModelFn(prompt, n) -> list[str]."
        ),
    )

    # -- Entailment adapter ----------------------------------------------------
    parser.add_argument(
        "--entailment-module",
        default=None,
        dest="entailment_module",
        metavar="MODULE:CALLABLE",
        help=(
            "Import an NLI entailment factory as 'module_path:callable_name'. "
            "The callable is invoked with no arguments and must return a "
            "NliFn(premise, hypothesis) -> str. If omitted, exact-match "
            "clustering is used."
        ),
    )

    # -- Kernel Language Entropy (KLE) ----------------------------------------
    parser.add_argument(
        "--kle-model",
        default=None,
        dest="kle_model",
        metavar="SENTENCE_TRANSFORMER_ID",
        help=(
            "If set, compute Kernel Language Entropy alongside the discrete "
            "scores using this sentence-transformer model "
            "(e.g. sentence-transformers/all-MiniLM-L6-v2). Requires the "
            "'kle' optional dependency group."
        ),
    )
    parser.add_argument(
        "--kle-kernel",
        default="rbf",
        choices=("rbf", "cosine"),
        dest="kle_kernel",
        help="Kernel function for KLE (default: rbf).",
    )
    parser.add_argument(
        "--kle-device",
        default="cpu",
        dest="kle_device",
        help=(
            "Device for the embedding model (default: cpu). KLE runs cheaply "
            "on CPU and keeps the GPU free for the generation model."
        ),
    )

    # -- RunConfig overrides ---------------------------------------------------
    parser.add_argument("--run-id", default="pipeline-001", dest="run_id")
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
        default="exact-match",
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

    # -- Resolve model ---------------------------------------------------------
    model_fn: ModelFn
    if args.stub_model or args.model_module is None:
        model_fn = _make_stub_model(seed=args.seed)
        if args.model == "synthetic-model":
            pass  # keep default
    else:
        try:
            factory = load_callable(args.model_module)
        except (ValueError, ImportError, AttributeError) as exc:
            parser.exit(status=1, message=f"error loading model module: {exc}\n")
        model_fn = factory()

    # -- Resolve entailment function -------------------------------------------
    entailment_fn: NliFn | None = None
    if args.entailment_module is not None:
        try:
            ent_factory = load_callable(args.entailment_module)
        except (ValueError, ImportError, AttributeError) as exc:
            parser.exit(
                status=1,
                message=f"error loading entailment module: {exc}\n",
            )
        entailment_fn = ent_factory()

    # -- Resolve KLE embedding function ---------------------------------------
    embedding_fn = None
    if args.kle_model is not None:
        try:
            from semantic_entropy.models import make_embedding_fn
            embedding_fn = make_embedding_fn(
                args.kle_model, device=args.kle_device,
            )
        except ImportError as exc:
            parser.exit(status=1, message=f"error loading embedding model: {exc}\n")

    # -- Load dataset ----------------------------------------------------------
    try:
        items = _load_items(
            args.dataset,
            data_path=args.data_path,
            limit=args.limit,
        )
    except DatasetError as exc:
        parser.exit(status=1, message=f"error loading dataset: {exc}\n")

    if not items:
        parser.exit(status=1, message="error: no prompt items loaded\n")

    # -- Build RunConfig -------------------------------------------------------
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

    # -- Run pipeline ----------------------------------------------------------
    try:
        n = run_pipeline(
            items, config, model_fn, args.output_jsonl,
            entailment_fn=entailment_fn,
            embedding_fn=embedding_fn,
            kle_kernel=args.kle_kernel,
        )
    except (SchemaError, ValueError, OSError) as exc:
        parser.exit(status=1, message=f"pipeline error: {exc}\n")

    print(f"wrote {n} scored records to {args.output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
