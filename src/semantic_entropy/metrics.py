"""Selective-prediction metrics for scored semantic-entropy records.

The metrics in this module evaluate whether higher uncertainty scores are
useful abstention signals. AUROC treats incorrect answers as the positive
class, so a score receives a high AUROC when incorrect records tend to have
higher uncertainty than correct records. AURAC is computed from the
rejection-accuracy curve obtained by rejecting the highest-uncertainty
records first and averaging retained-set accuracy over all non-empty
coverage levels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from semantic_entropy.cluster_check import check_cluster_consistency
from semantic_entropy.schema import SchemaError, validate_record

DEFAULT_SCORE_ORDER = (
    "naive_sample_entropy",
    "surface_entropy",
    "discrete_semantic_entropy",
    # Phase 2 extensions — included when present in the scored records.
    "kle",
    "probe_uncertainty",
)
MetricTableRow = dict[str, float | int | str | None]


def raw_accuracy(correctness_labels: Sequence[bool]) -> float:
    """Return the fraction of records marked correct."""

    _validate_correctness_labels(correctness_labels)
    return sum(correctness_labels) / len(correctness_labels)


def auroc(positive_labels: Sequence[bool], scores: Sequence[float]) -> float:
    """Return rank-based AUROC for scores where larger means more positive.

    Tied scores receive average ranks, so ties contribute 0.5 to the pairwise
    ranking probability. The positive class must be present alongside at least
    one negative example.
    """

    _validate_labels_and_scores(positive_labels, scores)
    positive_count = sum(positive_labels)
    negative_count = len(positive_labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("AUROC requires at least one positive and one negative label")

    ranks = _average_ranks(scores)
    positive_rank_sum = sum(
        rank for is_positive, rank in zip(positive_labels, ranks) if is_positive
    )
    return (positive_rank_sum - positive_count * (positive_count + 1) / 2) / (
        positive_count * negative_count
    )


def rejection_accuracy_curve(
    correctness_labels: Sequence[bool], uncertainty_scores: Sequence[float]
) -> list[dict[str, float | int]]:
    """Return retained accuracy after rejecting highest-uncertainty records.

    The first point keeps all records. Each following point rejects one more
    record, ordered from highest to lowest uncertainty, and reports accuracy
    on the retained non-empty set.
    """

    _validate_labels_and_scores(correctness_labels, uncertainty_scores)
    ordered = sorted(
        zip(uncertainty_scores, correctness_labels),
        key=lambda item: item[0],
        reverse=True,
    )
    total = len(ordered)
    retained_correct = sum(correctness_labels)
    points: list[dict[str, float | int]] = []

    for rejected_count in range(total):
        retained_count = total - rejected_count
        points.append(
            {
                "rejected": rejected_count,
                "retained": retained_count,
                "coverage": retained_count / total,
                "rejection_fraction": rejected_count / total,
                "accuracy": retained_correct / retained_count,
            }
        )
        if rejected_count < total - 1 and ordered[rejected_count][1]:
            retained_correct -= 1

    return points


def aurac(correctness_labels: Sequence[bool], uncertainty_scores: Sequence[float]) -> float:
    """Return stepwise area under the rejection-accuracy curve.

    This scaffold uses the average retained-set accuracy over all non-empty
    retention levels, equivalent to a step-function area with width ``1 / N``
    for each rejection step.
    """

    curve = rejection_accuracy_curve(correctness_labels, uncertainty_scores)
    return sum(point["accuracy"] for point in curve) / len(curve)


def aurac_paper(
    correctness_labels: Sequence[bool],
    uncertainty_scores: Sequence[float],
    *,
    n_quantiles: int = 20,
    lowest_retained: float = 0.1,
) -> float:
    """Return AURAC exactly as Farquhar et al. (2024) compute it.

    Their ``area_under_thresholded_accuracy`` in ``uncertainty/utils/eval_utils.py``::

        quantiles = np.linspace(0.1, 1, 20)
        select_accuracies = [accuracy_at_quantile(a, u, q) for q in quantiles]
        dx = quantiles[1] - quantiles[0]
        area = (select_accuracies * dx).sum()

    Three properties differ from :func:`aurac` and make the two incomparable:

    * the curve is sampled at 20 quantiles, not once per record;
    * it starts at 10 per cent retained, so the extreme tail is excluded;
    * it is a left Riemann sum weighted by ``dx = 0.9 / 19``, and is *not*
      normalised. A perfect detector scores ``20 * 0.9 / 19 = 0.947``, not 1.0,
      even though the paper describes the measure as increasing towards 1.

    Both are reported so that values in this thesis can be read against the
    original without rescaling.

    Args:
        correctness_labels: ``True`` where the answer is correct.
        uncertainty_scores: Higher means more likely wrong.
        n_quantiles: Number of retention levels sampled.
        lowest_retained: Smallest retained fraction.

    Returns:
        The unnormalised area, whose maximum is
        ``n_quantiles * (1 - lowest_retained) / (n_quantiles - 1)``.
    """
    _validate_labels_and_scores(correctness_labels, uncertainty_scores)
    if n_quantiles < 2:
        raise ValueError("n_quantiles must be at least 2")
    if not 0.0 < lowest_retained <= 1.0:
        raise ValueError("lowest_retained must lie in (0, 1]")

    accuracies = [bool(label) for label in correctness_labels]
    scores = list(uncertainty_scores)
    step = (1.0 - lowest_retained) / (n_quantiles - 1)
    quantiles = [lowest_retained + index * step for index in range(n_quantiles)]

    total = 0.0
    for quantile in quantiles:
        cutoff = _quantile(scores, quantile)
        retained = [
            accuracies[i] for i, score in enumerate(scores) if score <= cutoff
        ]
        if not retained:
            continue
        total += (sum(retained) / len(retained)) * step
    return total


def _quantile(values: Sequence[float], quantile: float) -> float:
    """Linear-interpolation quantile, matching ``numpy.quantile`` defaults."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_records(
    records: Sequence[Mapping[str, Any]], score_fields: Sequence[str] | None = None
) -> dict[str, Any]:
    """Summarize raw accuracy, AUROC, and AURAC for scored JSONL records."""

    _validate_records(records)
    correctness_labels = [record["correctness_label"] for record in records]
    fields = _resolve_score_fields(records, score_fields)
    score_metrics: dict[str, dict[str, float | None]] = {}
    coverage: dict[str, dict[str, int]] = {}

    for field in fields:
        labels, values, skipped = _score_pairs(records, field)
        incorrect = [not is_correct for is_correct in labels]
        try:
            auroc_value: float | None = auroc(incorrect, values)
        except ValueError:
            auroc_value = None
        score_metrics[field] = {
            "auroc": auroc_value,
            "aurac": aurac(labels, values),
            # Reported alongside so that any value quoted against Farquhar et
            # al. uses their definition. The two are not interchangeable: see
            # aurac_paper.
            "aurac_paper": aurac_paper(labels, values),
        }
        coverage[field] = {
            "scored": len(values),
            "degenerate_greedy_excluded": skipped,
        }

    summary: dict[str, Any] = {
        "num_records": len(records),
        "raw_accuracy": raw_accuracy(correctness_labels),
        "positive_class": "incorrect",
        "score_metrics": score_metrics,
    }
    if any(entry["degenerate_greedy_excluded"] for entry in coverage.values()):
        # Only emitted when a field actually has unscoreable prompts, so
        # artifacts without exclusions keep their previous shape exactly.
        summary["score_field_coverage"] = coverage
    return summary


def summarize_jsonl(input_path: Path, score_fields: Sequence[str] | None = None) -> dict[str, Any]:
    """Load scored JSONL records from ``input_path`` and summarize metrics."""

    records = _load_records(input_path)
    return summarize_records(records, score_fields)


def rejection_curve_rows(
    records: Sequence[Mapping[str, Any]], score_fields: Sequence[str] | None = None
) -> list[dict[str, float | int | str]]:
    """Return plot-ready rejection-accuracy rows for scored records."""

    _validate_records(records)
    fields = _resolve_score_fields(records, score_fields)
    rows: list[dict[str, float | int | str]] = []

    for field in fields:
        labels, values, _ = _score_pairs(records, field)
        for point in rejection_accuracy_curve(labels, values):
            rows.append({"score_field": field, **point})

    return rows


def metric_summary_rows(summary: Mapping[str, Any]) -> list[MetricTableRow]:
    """Return tabular rows comparing score-field metrics."""

    score_metrics = summary["score_metrics"]
    return [
        {
            "score_field": score_field,
            "num_records": summary["num_records"],
            "raw_accuracy": summary["raw_accuracy"],
            "positive_class": summary["positive_class"],
            "auroc": metrics["auroc"],
            "aurac": metrics["aurac"],
        }
        for score_field, metrics in score_metrics.items()
    ]


def write_metric_summary_table(
    summary: Mapping[str, Any],
    output_dir: Path = Path("results/tables"),
    basename: str = "metric_summary",
) -> dict[str, str]:
    """Write AUROC/AURAC score comparisons as small JSON and CSV tables."""

    rows = metric_summary_rows(summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "summary_table_json": str(output_dir / f"{basename}.json"),
        "summary_table_csv": str(output_dir / f"{basename}.csv"),
    }
    Path(artifacts["summary_table_json"]).write_text(
        json.dumps({"score_metrics": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_summary_table_csv(Path(artifacts["summary_table_csv"]), rows)
    return artifacts


def write_metric_artifacts(
    records: Sequence[Mapping[str, Any]],
    output_dir: Path,
    score_fields: Sequence[str] | None = None,
    summary: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Write summary and rejection-accuracy curves as small JSON/CSV files."""

    summary_payload = dict(summary or summarize_records(records, score_fields))
    rows = rejection_curve_rows(records, score_fields)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "summary_json": str(output_dir / "metrics_summary.json"),
        "curves_json": str(output_dir / "rejection_accuracy_curves.json"),
        "curves_csv": str(output_dir / "rejection_accuracy_curves.csv"),
    }

    summary_with_artifacts = {**summary_payload, "artifacts": artifacts}
    Path(artifacts["summary_json"]).write_text(
        json.dumps(summary_with_artifacts, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    Path(artifacts["curves_json"]).write_text(
        json.dumps({"curves": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_curve_csv(Path(artifacts["curves_csv"]), rows)
    return artifacts


def main(argv: Sequence[str] | None = None) -> int:
    """Run the scored-JSONL metrics CLI and return a process exit code."""

    parser = argparse.ArgumentParser(
        description=(
            "Compute raw accuracy, AUROC, and AURAC from scored semantic-entropy JSONL records."
        )
    )
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument(
        "--score-field",
        action="append",
        dest="score_fields",
        help="Score field to evaluate. May be supplied more than once.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional results directory for summary JSON and curve CSV/JSON artifacts.",
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        help="Optional results/tables directory for AUROC/AURAC summary table artifacts.",
    )
    parser.add_argument(
        "--table-basename",
        default="metric_summary",
        help="Output filename stem for summary table JSON and CSV artifacts.",
    )
    args = parser.parse_args(argv)

    try:
        records = _load_records(args.input_jsonl)
        summary = summarize_records(records, args.score_fields)
        artifacts: dict[str, str] = {}
        if args.output_dir is not None:
            artifacts.update(
                write_metric_artifacts(
                    records,
                    args.output_dir,
                    args.score_fields,
                    summary=summary,
                )
            )
        if args.table_dir is not None:
            artifacts.update(
                write_metric_summary_table(
                    summary,
                    args.table_dir,
                    basename=args.table_basename,
                )
            )
        if artifacts:
            summary["artifacts"] = artifacts
    except (OSError, SchemaError, ValueError) as error:
        parser.exit(status=1, message=f"error: {error}\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _validate_records(records: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        raise ValueError("at least one scored record is required")

    for index, record in enumerate(records, start=1):
        try:
            validate_record(record)
            check_cluster_consistency(record)
        except SchemaError as error:
            raise SchemaError(f"record {index}: {error}") from error


def _load_records(input_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                raise SchemaError(f"line {line_number}: empty JSONL record")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise SchemaError(f"line {line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(record, dict):
                raise SchemaError(f"line {line_number}: record must be a JSON object")
            records.append(record)
    return records


def _resolve_score_fields(
    records: Sequence[Mapping[str, Any]], requested_fields: Sequence[str] | None
) -> list[str]:
    if requested_fields:
        fields = list(dict.fromkeys(requested_fields))
    else:
        common_fields = set(records[0]["scores"])
        for record in records[1:]:
            common_fields &= set(record["scores"])
        fields = [field for field in DEFAULT_SCORE_ORDER if field in common_fields]
        fields.extend(sorted(common_fields - set(fields)))

    if not fields:
        raise ValueError("no score fields available for metric computation")
    return fields


def _score_pairs(
    records: Sequence[Mapping[str, Any]], field: str
) -> tuple[list[bool], list[float], int]:
    """Return (correctness labels, score values, skipped) for one score field.

    A record that carries no value for ``field`` is skipped ONLY when it
    declares a degenerate-greedy exclusion: such a prompt has no probe
    feature, so no probe score can exist for it, yet it remains part of the
    locked evaluation split. Any other missing score is an error, so a
    genuinely absent value can never be silently dropped from a metric.
    """

    labels: list[bool] = []
    values: list[float] = []
    skipped = 0
    for record in records:
        scores = record["scores"]
        if field not in scores:
            # Only a record with no usable feature at all may lack a probe
            # score. One whose feature was recovered declares feature_backfill
            # and must carry one, so a missing score there is a real error.
            from semantic_entropy.probes import sep_feature_unavailable

            if sep_feature_unavailable(record):
                skipped += 1
                continue
            prompt_id = record.get("prompt_id", "<unknown>")
            raise ValueError(f"record {prompt_id} is missing scores.{field}")
        labels.append(bool(record["correctness_label"]))
        values.append(_score_value(record, field))
    if not values:
        raise ValueError(f"no records carry scores.{field}")
    return labels, values, skipped


def _score_value(record: Mapping[str, Any], field: str) -> float:
    scores = record["scores"]
    if field not in scores:
        prompt_id = record.get("prompt_id", "<unknown>")
        raise ValueError(f"record {prompt_id} is missing scores.{field}")
    value = scores[field]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"scores.{field} must be numeric")
    if not math.isfinite(value):
        raise ValueError(f"scores.{field} must be finite")
    return float(value)


def _write_curve_csv(output_path: Path, rows: Sequence[Mapping[str, float | int | str]]) -> None:
    fieldnames = [
        "score_field",
        "rejected",
        "retained",
        "coverage",
        "rejection_fraction",
        "accuracy",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_table_csv(
    output_path: Path,
    rows: Sequence[Mapping[str, float | int | str | None]],
) -> None:
    fieldnames = [
        "score_field",
        "num_records",
        "raw_accuracy",
        "positive_class",
        "auroc",
        "aurac",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: "" if row[field] is None else row[field] for field in fieldnames}
            )


def _validate_labels_and_scores(labels: Sequence[bool], scores: Sequence[float]) -> None:
    _validate_correctness_labels(labels)
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    for score in scores:
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ValueError("scores must be numeric")
        if not math.isfinite(score):
            raise ValueError("scores must be finite")


def _validate_correctness_labels(labels: Sequence[bool]) -> None:
    if not labels:
        raise ValueError("labels must not be empty")
    if not all(isinstance(label, bool) for label in labels):
        raise ValueError("labels must be booleans")


def _average_ranks(scores: Sequence[float]) -> list[float]:
    indexed_scores = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0] * len(scores)
    start = 0

    while start < len(indexed_scores):
        end = start + 1
        while end < len(indexed_scores) and indexed_scores[end][1] == indexed_scores[start][1]:
            end += 1
        average_rank = (start + 1 + end) / 2
        for position in range(start, end):
            original_index = indexed_scores[position][0]
            ranks[original_index] = average_rank
        start = end

    return ranks


if __name__ == "__main__":
    raise SystemExit(main())
