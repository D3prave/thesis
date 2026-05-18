"""Train a Semantic Entropy Probe (SEP) from a scored JSONL file.

Reads a JSONL file produced by ``semantic-entropy-run-pipeline`` with
the ``--model-with-states-module`` adapter enabled (so every record
carries a ``hidden_states`` array), fits an L2-regularised logistic
regression probe, and writes the trained probe to a small JSON file.

Two training targets are supported via ``--label-source``:

* ``correctness`` (default) — predict ``not correctness_label``, the
  canonical SEP target.  This matches the AUROC convention used
  throughout the metrics module.
* ``semantic_entropy_threshold`` — predict whether an existing entropy
  score (e.g. ``discrete_semantic_entropy``) exceeds
  ``--label-threshold``.  Useful when you want the probe to mimic an
  entropy estimator rather than the correctness label.

Usage (after a generation run that collected hidden states)::

    uv run python scripts/train_sep_probe.py \\
        results/phase2/sep-train/scored.jsonl \\
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
        choices=["correctness", "semantic_entropy_threshold"],
        default="correctness",
        help=(
            "Training target. 'correctness' (default) predicts the "
            "positive class 'incorrect'. 'semantic_entropy_threshold' "
            "predicts whether the specified entropy score exceeds the "
            "threshold."
        ),
    )
    p.add_argument(
        "--source-score-field",
        default=None,
        help=(
            "Entropy score field to binarise when "
            "--label-source=semantic_entropy_threshold "
            "(e.g. discrete_semantic_entropy)."
        ),
    )
    p.add_argument(
        "--label-threshold",
        type=float,
        default=None,
        help=(
            "Threshold for binarising the entropy score. Pairs with "
            "--source-score-field."
        ),
    )
    p.add_argument(
        "--aggregate",
        choices=["first", "mean"],
        default="first",
        help=(
            "How to combine per-sample hidden states into a single "
            "training feature: 'first' (matches single-pass SEP "
            "inference) or 'mean' (denoised average across samples)."
        ),
    )
    p.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse L2 regularisation strength (default: 1.0).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed forwarded to scikit-learn (default: 0).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from semantic_entropy.probes import train_probe
    except ImportError as exc:
        sys.exit(f"ERROR: {exc}")

    records: list[dict[str, Any]] = []
    with args.scored_jsonl.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                sys.exit(
                    f"ERROR: {args.scored_jsonl}:{line_number}: invalid JSON: {exc}"
                )

    if not records:
        sys.exit(f"ERROR: no records found in {args.scored_jsonl}")

    print(f"Loaded {len(records)} records from {args.scored_jsonl}", flush=True)

    try:
        probe = train_probe(
            records,
            label_source=args.label_source,
            source_score_field=args.source_score_field,
            label_threshold=args.label_threshold,
            aggregate=args.aggregate,
            C=args.C,
            seed=args.seed,
        )
    except (ValueError, ImportError) as exc:
        sys.exit(f"ERROR: probe training failed: {exc}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    probe.save(args.output)
    print(
        f"Done. hidden_dim={probe.hidden_dim} train_size={probe.train_size} "
        f"positive_rate={probe.positive_rate:.3f} → {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
