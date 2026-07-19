"""Train a Semantic Entropy Probe (SEP) from a scored JSONL file.

Reads a JSONL file produced by ``semantic-entropy-run-pipeline`` with
the ``--model-with-states-module`` adapter enabled (so every record
carries a separate greedy ``sep_hidden_state``), fits an L2-regularized logistic
regression probe, and writes the trained probe to a small JSON file.

Two explicitly different probe artifacts are supported via ``--label-source``:

* ``semantic_entropy_threshold`` (default) — genuine SEP: derive the Eq. (5)
  threshold from the training records only and predict high semantic entropy.
  The train/evaluation records must share pinned question-conditioned
  DeBERTa-v3-large target lineage, contain no duplicate ``(dataset, prompt)``
  identities, and keep sampled max-token truncation at or below 1% per split.
* ``accuracy_probe_correctness`` — separate supervised baseline that predicts
  ``not correctness_label`` from the same greedy feature and split. It is
  serialized as an accuracy probe, never SEP.

Usage (after disjoint train/evaluation collection runs)::

    uv run python scripts/train_sep_probe.py \\
        results/phase2/sep-train/scored.jsonl \\
        --eval-jsonl results/phase2/sep-eval/scored.jsonl \\
        --output results/phase2/sep-train/probe.json

To then *score* a separate evaluation run with the trained probe::

    uv run python -m semantic_entropy.pipeline_cli \\
        --model-with-states-module semantic_entropy.adapters:make_phase1_model_with_states \\
        --sep-probe results/phase2/sep-train/probe.json \\
        --dataset triviaqa --data-path data/processed/triviaqa_val_500.jsonl \\
        results/phase2/sep-eval/scored.jsonl

Requirements:
    pip install -e '.[sep]'   (numpy + scikit-learn)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "scored_jsonl",
        type=Path,
        help=(
            "Path to a scored JSONL file produced with hidden-state "
            "collection enabled."
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        metavar="PROBE_JSON",
        help="Output path for the trained probe JSON artifact.",
    )
    p.add_argument(
        "--label-source",
        choices=["semantic_entropy_threshold", "accuracy_probe_correctness"],
        default="semantic_entropy_threshold",
        help=(
            "Training target. 'semantic_entropy_threshold' (default) creates "
            "a genuine SEP. 'accuracy_probe_correctness' creates the "
            "separately identified correctness baseline."
        ),
    )
    p.add_argument(
        "--source-score-field",
        default="discrete_semantic_entropy",
        help=(
            "Entropy score field to binarize when "
            "--label-source=semantic_entropy_threshold "
            "(e.g. discrete_semantic_entropy)."
        ),
    )
    p.add_argument(
        "--eval-jsonl",
        type=Path,
        default=None,
        help=(
            "Held-out evaluation JSONL. Required for both probe kinds so exact "
            "prompt-content disjointness and feature identity are audited. Its "
            "entropy values are never used to choose the SEP threshold."
        ),
    )
    p.add_argument(
        "--aggregate",
        choices=["first", "mean"],
        default="first",
        help=(
            "Feature aggregation. Canonical SEP and accuracy probes both "
            "require 'first' and use the singular greedy feature."
        ),
    )
    p.add_argument(
        "--pooling",
        choices=["last", "mean", "mean_last_k"],
        default="last",
        help=(
            "Feature pooling. Both canonical probes fix 'last', meaning the "
            "singular greedy response's final content token."
        ),
    )
    p.add_argument(
        "--mean-last-k",
        type=int,
        default=8,
        dest="mean_last_k",
        help="k for --pooling=mean_last_k (default: 8).",
    )
    p.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse L2 regularization strength (default: 1.0).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed forwarded to scikit-learn (default: 0).",
    )
    return p.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return records


def _prompt_ids(records: list[dict[str, Any]], role: str) -> set[str]:
    ids: set[str] = set()
    for index, record in enumerate(records):
        prompt_id = record.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise ValueError(f"{role} record {index}: prompt_id must be non-empty")
        if prompt_id in ids:
            raise ValueError(f"{role} records contain duplicate prompt_id={prompt_id!r}")
        ids.add(prompt_id)
    return ids


def _prompt_identities(
    records: list[dict[str, Any]], role: str
) -> set[tuple[str, str]]:
    identities: set[tuple[str, str]] = set()
    for index, record in enumerate(records):
        dataset = record.get("dataset")
        prompt = record.get("prompt")
        if not isinstance(dataset, str) or not dataset.strip():
            raise ValueError(f"{role} record {index}: dataset must be non-empty")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{role} record {index}: prompt must be non-empty")
        identity = (dataset, prompt)
        if identity in identities:
            raise ValueError(
                f"{role} records contain duplicate (dataset, prompt) identity"
            )
        identities.add(identity)
    return identities


def _sampled_truncation_audit(
    records: list[dict[str, Any]], role: str
) -> dict[str, Any]:
    total = 0
    truncated = 0
    for index, record in enumerate(records):
        feature_metadata = record.get("sep_feature_metadata")
        if not isinstance(feature_metadata, dict):
            raise ValueError(
                f"{role} record {index}: sep_feature_metadata must be a mapping"
            )
        record_total = feature_metadata.get("sampling_num_responses")
        record_truncated = feature_metadata.get("sampled_max_new_tokens_count")
        record_terminated = feature_metadata.get("sampled_eos_or_eot_count")
        for field, value in (
            ("sampling_num_responses", record_total),
            ("sampled_max_new_tokens_count", record_truncated),
            ("sampled_eos_or_eot_count", record_terminated),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"{role} record {index}: sep_feature_metadata.{field} must "
                    "be a non-negative integer"
                )
        if record_total <= 0 or record_truncated + record_terminated != record_total:
            raise ValueError(
                f"{role} record {index}: sampled termination counts must sum to "
                "a positive sampling_num_responses"
            )
        total += record_total
        truncated += record_truncated
    rate = truncated / total
    if truncated * 100 > total:
        raise ValueError(
            f"{role} sampled max-token truncation rate {rate:.6f} exceeds the "
            "1% limit"
        )
    return {
        f"{role}_sampled_response_count": total,
        f"{role}_sampled_max_new_tokens_count": truncated,
        f"{role}_sampled_truncation_rate": rate,
    }


def split_audit_metadata(
    train_records: list[dict[str, Any]], eval_records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Verify target lineage, leakage, and truncation; return split provenance."""
    from semantic_entropy.probes import (
        CANONICAL_SEP_MAX_SAMPLED_TRUNCATION_RATE,
        canonical_entropy_target_provenance,
        canonical_shared_sep_cell_identity,
        prompt_content_sha256,
    )

    train_ids = _prompt_ids(train_records, "training")
    eval_ids = _prompt_ids(eval_records, "evaluation")
    overlap = sorted(train_ids & eval_ids)
    if overlap:
        preview = ", ".join(overlap[:5])
        raise ValueError(
            f"training/evaluation prompt overlap is nonzero ({len(overlap)}): "
            f"{preview}"
        )

    train_identities = _prompt_identities(train_records, "training")
    eval_identities = _prompt_identities(eval_records, "evaluation")
    content_overlap = sorted(train_identities & eval_identities)
    if content_overlap:
        preview = ", ".join(
            f"{dataset}:{prompt[:80]}" for dataset, prompt in content_overlap[:5]
        )
        raise ValueError(
            "training/evaluation (dataset, prompt) content overlap is nonzero "
            f"({len(content_overlap)}): {preview}"
        )

    def digest(ids: set[str]) -> str:
        payload = "\n".join(sorted(ids)).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def identity_digest(identities: set[tuple[str, str]]) -> str:
        payload = json.dumps(
            sorted([list(identity) for identity in identities]),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    entropy_target_provenance = canonical_entropy_target_provenance(
        train_records, eval_records
    )
    cell_identity = canonical_shared_sep_cell_identity(
        train_records, eval_records
    )

    return {
        "probe_fit_scope": "training_records_only",
        "threshold_fit_scope": "training_records_only",
        "split_overlap_count": 0,
        "split_content_overlap_count": 0,
        "train_prompt_count": len(train_ids),
        "eval_prompt_count": len(eval_ids),
        "train_prompt_ids_sha256": digest(train_ids),
        "eval_prompt_ids_sha256": digest(eval_ids),
        "train_prompt_content_sha256": prompt_content_sha256(train_records),
        "eval_prompt_content_sha256": prompt_content_sha256(eval_records),
        "train_prompt_identity_sha256": identity_digest(train_identities),
        "eval_prompt_identity_sha256": identity_digest(eval_identities),
        "analysis_code_commit": entropy_target_provenance[
            "analysis_code_commit"
        ],
        "cell_identity": cell_identity,
        "entropy_target_provenance": entropy_target_provenance,
        "sampled_truncation_rate_limit": (
            CANONICAL_SEP_MAX_SAMPLED_TRUNCATION_RATE
        ),
        **_sampled_truncation_audit(train_records, "train"),
        **_sampled_truncation_audit(eval_records, "eval"),
    }


def main() -> None:
    args = parse_args()

    try:
        from semantic_entropy.probes import (
            train_probe,
            validate_canonical_accuracy_scoring_records,
            validate_canonical_sep_scoring_records,
        )
    except ImportError as exc:
        sys.exit(f"ERROR: {exc}")

    try:
        records = _load_jsonl(args.scored_jsonl)
    except (OSError, ValueError) as exc:
        sys.exit(f"ERROR: {exc}")

    if not records:
        sys.exit(f"ERROR: no records found in {args.scored_jsonl}")

    print(f"Loaded {len(records)} records from {args.scored_jsonl}", flush=True)

    internal_label_source = (
        "semantic_entropy_threshold"
        if args.label_source == "semantic_entropy_threshold"
        else "correctness"
    )
    split_metadata: dict[str, Any] = {}
    eval_records: list[dict[str, Any]] | None = None
    if args.eval_jsonl is None:
        sys.exit(
            "ERROR: canonical SEP and accuracy-probe training require "
            "--eval-jsonl so the held-out split can be audited"
        )
    if args.aggregate != "first" or args.pooling != "last":
        sys.exit(
            "ERROR: canonical probes require --aggregate=first and "
            "--pooling=last for the singular greedy feature"
        )
    if (
        internal_label_source == "semantic_entropy_threshold"
        and args.source_score_field is None
    ):
        sys.exit("ERROR: genuine SEP training requires --source-score-field")
    try:
        eval_records = _load_jsonl(args.eval_jsonl)
        if not eval_records:
            raise ValueError(f"no records found in {args.eval_jsonl}")

        # Restrict both splits to prompts whose greedy feature is present.
        # Degenerate-greedy prompts carry no probe feature: they are not probe
        # training examples and are not probe-scored, so the audited prompt
        # counts must match the probe's fitted train_size and the scored eval
        # count. They remain in the collection file and the entropy analysis.
        def _feature_present(record):
            metadata = record.get("sep_feature_metadata") or {}
            return metadata.get("greedy_degenerate_excluded") is not True

        n_train_total = len(records)
        n_eval_total = len(eval_records)
        records = [record for record in records if _feature_present(record)]
        eval_records = [
            record for record in eval_records if _feature_present(record)
        ]
        if not records:
            raise ValueError("all training records had excluded greedy features")
        if not eval_records:
            raise ValueError("all evaluation records had excluded greedy features")
        if n_train_total != len(records) or n_eval_total != len(eval_records):
            print(
                "Excluded degenerate-greedy prompts: "
                f"train {n_train_total - len(records)}, "
                f"eval {n_eval_total - len(eval_records)}",
                flush=True,
            )
        split_metadata = split_audit_metadata(records, eval_records)
    except (OSError, ValueError) as exc:
        sys.exit(f"ERROR: split audit failed: {exc}")

    try:
        probe = train_probe(
            records,
            label_source=internal_label_source,
            source_score_field=(
                args.source_score_field
                if internal_label_source == "semantic_entropy_threshold"
                else None
            ),
            aggregate=args.aggregate,
            pooling=args.pooling,
            mean_last_k=args.mean_last_k,
            C=args.C,
            seed=args.seed,
            metadata=split_metadata,
        )
    except (ValueError, ImportError) as exc:
        sys.exit(f"ERROR: probe training failed: {exc}")

    if eval_records is None:  # defensive: required above
        sys.exit("ERROR: held-out evaluation records were not loaded")
    # Drop held-out prompts whose greedy feature was excluded (degeneration):
    # they carry no probe feature and are validated/scored out of the probe
    # evaluation. They remain in the collection file and the entropy analysis.
    eval_records_present = [
        record
        for record in eval_records
        if (record.get("sep_feature_metadata") or {}).get(
            "greedy_degenerate_excluded"
        )
        is not True
    ]
    n_excluded_eval = len(eval_records) - len(eval_records_present)
    if n_excluded_eval:
        print(
            f"Excluding {n_excluded_eval} held-out record(s) with degenerate "
            f"greedy feature from probe validation",
            flush=True,
        )
    try:
        if internal_label_source == "semantic_entropy_threshold":
            validate_canonical_sep_scoring_records(eval_records_present, probe)
        else:
            validate_canonical_accuracy_scoring_records(eval_records_present, probe)
    except ValueError as exc:
        sys.exit(f"ERROR: canonical probe artifact validation failed: {exc}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    probe.save(args.output)
    print(
        f"Done. hidden_dim={probe.hidden_dim} train_size={probe.train_size} "
        f"positive_rate={probe.positive_rate:.3f} pooling={probe.pooling} → {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
