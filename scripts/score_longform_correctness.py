"""Score long-form bio records with an LLM judge and fill correctness fields.

Reads a scored JSONL file produced by the bio pipeline, loads the LLM judge,
fills ``correctness_label`` (bool) and optionally
``scores.correctness_score`` (float in [0,1]) for each record, and writes a
new JSONL alongside. The script is idempotent by default: records that
already have a non-placeholder correctness_label are re-graded unconditionally
(the pipeline writes False as a placeholder so we can't distinguish it from a
genuine False without a separate flag; --skip-already-true skips records
where correctness_label is True).

Usage::

    uv run python scripts/score_longform_correctness.py \\
        results/phase2/<run>/scored.jsonl \\
        results/phase2/<run>/scored_graded.jsonl \\
        --judge-model Qwen/Qwen2.5-72B-Instruct \\
        --tensor-parallel-size 4
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from semantic_entropy.cluster_check import check_cluster_consistency
from semantic_entropy.longform_correctness import grade_records_inplace
from semantic_entropy.schema import SchemaError, validate_record


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input_jsonl", type=Path, help="Scored JSONL from bio pipeline.")
    p.add_argument(
        "output_jsonl",
        type=Path,
        help="Output JSONL with correctness_label and scores.correctness_score filled.",
    )
    p.add_argument(
        "--judge-model",
        default=None,
        help="HuggingFace judge model ID (default: SE_JUDGE_MODEL or Qwen/Qwen2.5-72B-Instruct).",
    )
    p.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        dest="tensor_parallel_size",
        help="vLLM tensor-parallel degree (default: SE_TENSOR_PARALLEL_SIZE or 4).",
    )
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        dest="gpu_memory_utilization",
        help="vLLM GPU memory utilization (default: 0.90).",
    )
    p.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        dest="max_model_len",
        help="vLLM max_model_len cap.",
    )
    p.add_argument(
        "--dtype",
        default=None,
        help="vLLM dtype (default: auto).",
    )
    p.add_argument(
        "--enforce-eager",
        action="store_true",
        default=None,
        dest="enforce_eager",
        help="Pass enforce_eager=True to vLLM.",
    )
    p.add_argument(
        "--no-confidence",
        action="store_true",
        default=False,
        dest="no_confidence",
        help="Skip the confidence follow-up question (faster, no correctness_score).",
    )
    p.add_argument(
        "--dataset-filter",
        default="bio",
        dest="dataset_filter",
        help=(
            "Only grade records whose dataset field matches this value. "
            "Use 'all' to grade every record (default: bio)."
        ),
    )
    return p.parse_args(argv)


def _resolve(value, env_key: str, default):
    if value is not None:
        return value
    raw = os.environ.get(env_key)
    if raw is not None:
        return type(default)(raw)
    return default


def _resolve_enforce_eager(flag) -> bool:
    if flag is not None:
        return bool(flag)
    raw = os.environ.get("SE_VLLM_ENFORCE_EAGER")
    if raw is None:
        return True
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.input_jsonl.resolve() == args.output_jsonl.resolve():
        raise ValueError("input_jsonl and output_jsonl must be different files")

    # ------------------------------------------------------------------
    # Load records
    # ------------------------------------------------------------------
    records: list[dict[str, Any]] = []
    with args.input_jsonl.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SchemaError(
                    f"line {line_number}: invalid JSON: {exc}"
                ) from exc
            records.append(record)

    if not records:
        raise ValueError(f"No records found in {args.input_jsonl}")

    print(f"Loaded {len(records)} records from {args.input_jsonl}", flush=True)

    # Filter to bio records (or all if requested).
    if args.dataset_filter != "all":
        to_grade = [r for r in records if r.get("dataset") == args.dataset_filter]
        skip = [r for r in records if r.get("dataset") != args.dataset_filter]
        print(
            f"Grading {len(to_grade)} '{args.dataset_filter}' records "
            f"(skipping {len(skip)} others).",
            flush=True,
        )
    else:
        to_grade = records
        skip = []

    # ------------------------------------------------------------------
    # Load judge
    # ------------------------------------------------------------------
    judge_model = (
        args.judge_model
        or os.environ.get("SE_JUDGE_MODEL")
        or "Qwen/Qwen2.5-72B-Instruct"
    )

    from semantic_entropy.llm_judge import BatchedLlmJudge

    judge = BatchedLlmJudge(
        model_name=judge_model,
        tensor_parallel_size=_resolve(
            args.tensor_parallel_size, "SE_TENSOR_PARALLEL_SIZE", 4
        ),
        gpu_memory_utilization=_resolve(
            args.gpu_memory_utilization, "SE_GPU_MEMORY_UTILIZATION", 0.90
        ),
        dtype=args.dtype or os.environ.get("SE_VLLM_DTYPE", "auto"),
        max_model_len=args.max_model_len or None,
        enforce_eager=_resolve_enforce_eager(args.enforce_eager),
    )

    # ------------------------------------------------------------------
    # Grade
    # ------------------------------------------------------------------
    grade_records_inplace(
        to_grade,
        judge,
        include_confidence=not args.no_confidence,
    )

    # ------------------------------------------------------------------
    # Write output — merge graded + skipped in original order
    # ------------------------------------------------------------------
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    graded_by_id = {r.get("prompt_id"): r for r in to_grade}
    merged: list[dict[str, Any]] = []
    for record in records:
        pid = record.get("prompt_id")
        merged.append(graded_by_id.get(pid, record))

    with args.output_jsonl.open("w", encoding="utf-8") as fh:
        for record in merged:
            try:
                validate_record(record)
                check_cluster_consistency(record)
            except SchemaError as exc:
                print(
                    f"  WARNING: record {record.get('prompt_id')} failed validation: {exc}",
                    flush=True,
                )
            fh.write(json.dumps(record, sort_keys=True) + "\n")

    print(
        f"Wrote {len(merged)} records to {args.output_jsonl} "
        f"({len(to_grade)} graded, {len(skip)} passed through).",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
