"""Re-run semantic clustering on an existing scored JSONL file.

The script reuses saved ``sampled_answers`` and ``normalized_answers``. It
does not call the generation model. It updates semantic clusters,
representatives, core entropy scores, and the ``entailment_backend`` label.
Optional extension scores such as ``kle`` are preserved when already present.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from semantic_entropy.cluster_check import check_cluster_consistency
from semantic_entropy.clustering import NliFn, exact_match_entailment_fn, nli_cluster
from semantic_entropy.pipeline_cli import load_callable
from semantic_entropy.schema import SchemaError, validate_record
from semantic_entropy.scoring import score_record


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument(
        "--entailment-module",
        default="semantic_entropy.adapters:make_nli",
        metavar="MODULE:CALLABLE",
        help="Factory returning an NliFn. Ignored with --exact-match.",
    )
    parser.add_argument(
        "--entailment-backend",
        default=None,
        help="Label to write into each output record.",
    )
    parser.add_argument(
        "--exact-match",
        action="store_true",
        help="Use exact-match entailment for local smoke tests.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run_id to write into each output record.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _ensure_distinct_paths(args.input_jsonl, args.output_jsonl)

    entailment_fn = _make_entailment_fn(args)
    backend = args.entailment_backend or _default_backend(args.exact_match)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with args.input_jsonl.open("r", encoding="utf-8") as source:
        with args.output_jsonl.open("w", encoding="utf-8") as target:
            for line_number, line in enumerate(source, start=1):
                record = _load_record(line, line_number)
                validate_record(record)

                clusters, representatives = nli_cluster(
                    record["normalized_answers"],
                    entailment_fn,
                )
                preserved_scores = {
                    key: value
                    for key, value in record["scores"].items()
                    if key
                    not in {
                        "naive_sample_entropy",
                        "surface_entropy",
                        "discrete_semantic_entropy",
                    }
                }
                record["semantic_clusters"] = clusters
                record["cluster_representatives"] = representatives
                record["scores"] = score_record(
                    record["sampled_answers"],
                    record["normalized_answers"],
                    clusters,
                )
                record["scores"].update(preserved_scores)
                record["entailment_backend"] = backend
                if args.run_id is not None:
                    record["run_id"] = args.run_id

                validate_record(record)
                check_cluster_consistency(record)
                target.write(json.dumps(record, sort_keys=True) + "\n")
                count += 1

    print(f"wrote {count} reclustered records to {args.output_jsonl}")
    return 0


def _make_entailment_fn(args: argparse.Namespace) -> NliFn:
    if args.exact_match:
        return exact_match_entailment_fn
    factory = load_callable(args.entailment_module)
    return factory()


def _default_backend(exact_match: bool) -> str:
    if exact_match:
        return "exact-match"
    model_name = os.environ.get("SE_NLI_MODEL", "cross-encoder/nli-deberta-v3-base")
    return model_name.rsplit("/", maxsplit=1)[-1]


def _load_record(line: str, line_number: int) -> dict[str, Any]:
    if not line.strip():
        raise SchemaError(f"line {line_number}: empty JSONL record")
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        raise SchemaError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
    if not isinstance(record, dict):
        raise SchemaError(f"line {line_number}: record must be a JSON object")
    return record


def _ensure_distinct_paths(input_path: Path, output_path: Path) -> None:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input_jsonl and output_jsonl must be different files")


if __name__ == "__main__":
    raise SystemExit(main())
