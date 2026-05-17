"""Command-line entry point for scoring synthetic JSONL prompt records.

This tool consumes a JSONL file whose records already contain
``normalized_answers`` and ``semantic_clusters`` -- i.e. the upstream
synthetic clustering step has already been performed, or, in the smoke-test
case, the labels are hand-authored -- computes the two Phase 1 uncertainty
scores defined in :mod:`semantic_entropy.scoring`, validates the resulting
record against the experimental JSONL schema and cluster-consistency checks,
and writes one canonicalized JSON line per record. The CLI is intentionally
strict: empty input lines, JSON-decode failures, and validation violations
are reported with their source line number and abort the run, so that
downstream metric code never sees a partially valid dataset.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from semantic_entropy.cluster_check import check_cluster_consistency
from semantic_entropy.schema import SchemaError, validate_record
from semantic_entropy.scoring import score_record


def score_jsonl(input_path: Path, output_path: Path) -> int:
    """Score every record in ``input_path`` and write canonicalized JSONL.

    For each non-empty input line the function decodes the JSON object,
    populates the ``scores`` field by calling
    :func:`semantic_entropy.scoring.score_record`, validates the augmented
    record against the experimental schema and cluster-consistency checks,
    and emits a single line with sorted keys for byte-stable, diff-friendly
    output. The number of successfully written records is returned.
    Identical input and output paths are rejected up front, since the
    read-then-overwrite loop would otherwise truncate the source.
    """

    if input_path.resolve() == output_path.resolve():
        raise ValueError("input_jsonl and output_jsonl must be different paths")

    records_written = 0
    with input_path.open("r", encoding="utf-8") as input_file, output_path.open(
        "w", encoding="utf-8"
    ) as output_file:
        for line_number, line in enumerate(input_file, start=1):
            record = _load_record(line, line_number)
            _add_scores(record, line_number)
            try:
                validate_record(record)
                check_cluster_consistency(record)
            except SchemaError as error:
                raise SchemaError(f"line {line_number}: {error}") from error
            output_file.write(json.dumps(record, sort_keys=True) + "\n")
            records_written += 1

    return records_written


def main(argv: Sequence[str] | None = None) -> int:
    """Run the synthetic JSONL scoring CLI and return a process exit code."""

    parser = argparse.ArgumentParser(
        description=(
            "Score synthetic semantic-entropy JSONL records that already "
            "carry normalized answers and semantic cluster identifiers. "
            "This tool is a Phase 1 smoke utility; it does not perform "
            "model inference, dataset loading, or entailment clustering."
        )
    )
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    args = parser.parse_args(argv)

    try:
        records_written = score_jsonl(args.input_jsonl, args.output_jsonl)
    except (OSError, SchemaError, ValueError) as error:
        parser.exit(status=1, message=f"error: {error}\n")

    print(f"wrote {records_written} records to {args.output_jsonl}")
    return 0


def _load_record(line: str, line_number: int) -> dict[str, Any]:
    if not line.strip():
        raise SchemaError(f"line {line_number}: empty JSONL record")

    try:
        record = json.loads(line)
    except json.JSONDecodeError as error:
        raise SchemaError(f"line {line_number}: invalid JSON: {error.msg}") from error

    if not isinstance(record, dict):
        raise SchemaError(f"line {line_number}: record must be a JSON object")

    return record


def _add_scores(record: dict[str, Any], line_number: int) -> None:
    try:
        normalized_answers = record["normalized_answers"]
        semantic_clusters = record["semantic_clusters"]
    except KeyError as error:
        raise SchemaError(
            f"line {line_number}: {error.args[0]} is required for scoring"
        ) from error

    try:
        record["scores"] = score_record(normalized_answers, semantic_clusters)
    except (TypeError, ValueError) as error:
        raise SchemaError(f"line {line_number}: {error}") from error


if __name__ == "__main__":
    raise SystemExit(main())
