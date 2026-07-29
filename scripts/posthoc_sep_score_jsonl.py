#!/usr/bin/env python3
"""Post-hoc SEP scoring: apply a trained probe to an existing scored JSONL.

Reads a scored JSONL that already has the separate greedy ``sep_hidden_state``
(produced by a phase2 SEP collect job), applies a trained probe, writes a new
JSONL with its explicitly named uncertainty score, and recomputes metrics.

Usage::

    uv run python scripts/posthoc_sep_score_jsonl.py \\
        results/phase2/<EVAL_RUN>/scored.jsonl \\
        results/phase2/<EVAL_RUN>/scored_sep.jsonl \\
        --probe results/phase2/<TRAIN_RUN>/probe.json

For the distinct correctness-trained baseline, pass
``--expected-probe-kind accuracy_probe``. That mode writes
``scores.accuracy_probe_uncertainty`` and ``accuracy_probe_provenance``; it
never writes either SEP field.

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
    parser.add_argument("input_jsonl", help="Scored JSONL with sep_hidden_state.")
    parser.add_argument("output_jsonl", help="Output path for probe-scored JSONL.")
    parser.add_argument(
        "--probe",
        required=True,
        metavar="PROBE_JSON",
        help="Path to trained probe.json produced by train_sep_probe.py.",
    )
    parser.add_argument(
        "--expected-probe-kind",
        choices=["semantic_entropy_probe", "accuracy_probe"],
        default="semantic_entropy_probe",
        help=(
            "Artifact kind to accept. Accuracy probes require an explicit value "
            "and are written to separately named score/provenance fields."
        ),
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

    if input_path.resolve() == output_path.resolve():
        print("ERROR: input_jsonl and output_jsonl must differ", file=sys.stderr)
        sys.exit(2)

    from semantic_entropy.probes import (
        SEPProbe,
        canonical_accuracy_probe_provenance,
        canonical_sep_provenance,
        score_probe_for_record,
        validate_canonical_accuracy_scoring_records,
        validate_canonical_sep_scoring_records,
    )

    try:
        probe = SEPProbe.load(probe_path)
        if probe.effective_probe_kind != args.expected_probe_kind:
            raise ValueError(
                f"artifact kind {probe.effective_probe_kind!r} does not match "
                f"--expected-probe-kind={args.expected_probe_kind!r}"
            )
        records = []
        with input_path.open(encoding="utf-8") as fin:
            for line_number, line in enumerate(fin, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{input_path}:{line_number}: invalid JSON: {exc}"
                    ) from exc
        if not records:
            raise ValueError(f"no records found in {input_path}")
        # Prompts whose greedy feature was excluded (degeneration) carry no
        # probe feature and cannot be probe-scored. The probe's audited split
        # is exactly the feature-present subset (see train_sep_probe's
        # split_audit_metadata), so the validation and the written artifact
        # both use that subset. The excluded prompts remain in the collection
        # and q-conditioned artifacts and in the entropy analysis; the
        # completion seal reconstructs the locked split as present + excluded.
        n_excluded = sum(
            1
            for record in records
            if (record.get("sep_feature_metadata") or {}).get(
                "greedy_degenerate_excluded"
            )
            is True
        )
        if n_excluded >= len(records):
            raise ValueError(
                f"all records in {input_path} had excluded greedy features"
            )
        if n_excluded:
            print(
                f"Excluding {n_excluded} record(s) with degenerate greedy "
                "feature from probe scoring"
            )
        records = [
            record
            for record in records
            if (record.get("sep_feature_metadata") or {}).get(
                "greedy_degenerate_excluded"
            )
            is not True
        ]
        if args.expected_probe_kind == "semantic_entropy_probe":
            validate_canonical_sep_scoring_records(records, probe)
            provenance = canonical_sep_provenance(probe)
            score_field = "probe_uncertainty"
            provenance_field = "probe_provenance"
        else:
            validate_canonical_accuracy_scoring_records(records, probe)
            provenance = canonical_accuracy_probe_provenance(probe)
            score_field = "accuracy_probe_uncertainty"
            provenance_field = "accuracy_probe_provenance"
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(
            f"ERROR: canonical SEP validation failed: {exc}",
            file=sys.stderr,
        )
        sys.exit(2)
    print(
        f"Loaded {probe.effective_probe_kind}: hidden_dim={probe.hidden_dim} "
        f"pooling={probe.pooling} "
        f"train_size={probe.train_size} positive_rate={probe.positive_rate:.3f}"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    n_skipped = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for record in records:
            score = score_probe_for_record(record, probe)
            if "scores" not in record or record["scores"] is None:
                record["scores"] = {}
            if score is None:
                # No probe feature exists for this prompt, so no probe score
                # can exist and the prompt is not part of what the probe
                # evaluates: it is omitted from the probe-scored artifact
                # (the convention the sealed v2 artifacts and the completion
                # seal use). It remains in the collection and q-conditioned
                # artifacts and in the entropy analysis. A scoreless prompt
                # must be attributable to the declared degeneration exclusion
                # and nothing else.
                excluded = (
                    record.get("sep_feature_metadata") or {}
                ).get("greedy_degenerate_excluded") is True
                if not excluded:
                    print(
                        "ERROR: record "
                        f"{record.get('prompt_id', '<unknown>')!r} produced no "
                        f"{score_field} but is not a declared degenerate-greedy "
                        "exclusion",
                        file=sys.stderr,
                    )
                    sys.exit(2)
                n_skipped += 1
                continue
            record["scores"][score_field] = score
            record[provenance_field] = provenance
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
            if n % 25 == 0:
                print(f"  scored {n} records")

    print(f"wrote {n} SEP-scored records to {output_path}")
    if n_skipped:
        print(
            f"omitted {n_skipped} degenerate-greedy record(s) with no probe "
            "feature (present + excluded reconstructs the locked split)"
        )


if __name__ == "__main__":
    main()
