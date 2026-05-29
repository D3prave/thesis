#!/usr/bin/env python3
"""Post-hoc SEP scoring: apply a trained probe to an existing scored JSONL.

Reads a scored JSONL that already has ``hidden_states`` (produced by a
phase2 SEP collect job), applies a trained SEP probe, writes a new JSONL
with ``scores.probe_uncertainty`` filled in, and recomputes metrics.

Usage::

    uv run python scripts/posthoc_sep_score_jsonl.py \\
        results/phase2/<EVAL_RUN>/scored.jsonl \\
        results/phase2/<EVAL_RUN>/scored_sep.jsonl \\
        --probe results/phase2/<TRAIN_RUN>/probe.json

    # Then recompute metrics:
    uv run python -m semantic_entropy.metrics \\
        results/phase2/<EVAL_RUN>/scored_sep.jsonl \\
        --output-dir results/phase2/<EVAL_RUN>/metrics_sep
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply a trained SEP probe to an existing scored JSONL."
    )
    parser.add_argument("input_jsonl", help="Scored JSONL with hidden_states.")
    parser.add_argument("output_jsonl", help="Output path for probe-scored JSONL.")
    parser.add_argument(
        "--probe",
        required=True,
        metavar="PROBE_JSON",
        help="Path to trained probe.json produced by train_sep_probe.py.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    probe_path = Path(args.probe)

    if not input_path.exists():
        print(f"ERROR: input not found: {input_path}", file=sys.stderr)
        sys.exit(2)
    if not probe_path.exists():
        print(f"ERROR: probe not found: {probe_path}", file=sys.stderr)
        sys.exit(2)

    from semantic_entropy.probes import SEPProbe, score_probe_for_record

    probe = SEPProbe.load(probe_path)
    print(
        f"Loaded probe: hidden_dim={probe.hidden_dim} pooling={probe.pooling} "
        f"train_size={probe.train_size} positive_rate={probe.positive_rate:.3f}"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with input_path.open(encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            score = score_probe_for_record(record, probe)
            if "scores" not in record or record["scores"] is None:
                record["scores"] = {}
            record["scores"]["probe_uncertainty"] = score
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
            if n % 25 == 0:
                print(f"  scored {n} records")

    print(f"wrote {n} SEP-scored records to {output_path}")


if __name__ == "__main__":
    main()
