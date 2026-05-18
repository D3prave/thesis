"""Structural validation of semantic-entropy JSONL prompt records.

A prompt record is the unit of analysis used throughout Phase 1 and Phase 2:
one record per prompt, written as a single line of UTF-8 JSON. The canonical
schema is specified in ``docs/experiment_protocol.md`` and mirrored in
Chapter 4 of the thesis. This module enforces *structural* conformance only:
required fields are present, value types are correct, enumerated fields take
admissible values, and the cardinalities of the per-sample arrays
(``sampled_answers``, ``normalized_answers``, ``semantic_clusters``) agree
with ``decoding.num_samples``.

Two classes of check are deliberately out of scope:

* *Numerical range constraints* on decoding hyperparameters
  (``temperature``, ``top_p``). The protocol fixes default values, but
  Phase 2 includes a decoding-temperature sensitivity sweep, so enforcing
  ranges here would obstruct legitimate ablations.

* *Semantic consistency* of the clustering output -- canonical, contiguous,
  non-negative cluster identifiers and one-to-one correspondence between
  cluster IDs and representatives. These properties are stronger than the
  shape constraints checked here and are validated by the dedicated cluster
  utility (Phase 1 follow-on work).

Schema violations are reported via :class:`SchemaError` with a precise field
path, so that the CLI scoring pipeline can attribute failures to a specific
line of input.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


class SchemaError(ValueError):
    """Raised when a prompt record violates the experimental JSONL schema."""


REQUIRED_FIELDS = {
    "run_id",
    "phase",
    "cluster",
    "prompt_id",
    "dataset",
    "split",
    "prompt",
    "reference_answers",
    "model",
    "model_tier",
    "decoding",
    "sampled_answers",
    "normalized_answers",
    "semantic_clusters",
    "cluster_representatives",
    "scores",
    "entailment_backend",
    "correctness_label",
}

PHASES = {"phase1", "phase2"}
CLUSTERS = {"local", "tinygpu", "alex", "helma"}
MODEL_TIERS = {"small", "7b_8b", "70b_plus"}
SCORE_FIELDS = {"surface_entropy", "discrete_semantic_entropy"}
# Optional score fields produced by Phase 2 extensions.  These are
# validated when present but their absence is not an error.
OPTIONAL_SCORE_FIELDS = {
    "naive_sample_entropy",
    # Kernel Language Entropy (kle.py) — Nikitin et al. (2024).
    "kle",
    # Semantic Entropy Probe (probes.py) score — Slobodkin et al. (2023).
    # Stored as a probability of "uncertain" in [0, 1]; finite and
    # non-negative, so the same validation rule applies.
    "probe_uncertainty",
}


def validate_record(record: Mapping[str, Any]) -> None:
    """Validate a single prompt record against the experimental schema.

    The check is structural and fails fast on the first violation found.
    Required keys must be present; string-typed fields, enumerated fields
    (``phase``, ``cluster``, ``model_tier``), and the ``reference_answers``,
    ``sampled_answers``, ``normalized_answers``, and
    ``cluster_representatives`` lists are type-checked; the ``decoding``
    sub-object must specify a positive ``num_samples``, finite
    ``temperature`` and ``top_p``, a positive ``max_new_tokens``, and an
    integer ``seed``. The lengths of all per-sample arrays must equal
    ``decoding.num_samples``, and the number of cluster representatives must
    equal the number of *distinct* cluster identifiers observed. The two
    score fields are required to be finite and non-negative, which is the
    range guaranteed by the entropy estimators in
    :mod:`semantic_entropy.scoring`.

    Non-required ``scores`` keys are permitted: this allows Phase 2 extension
    metrics (e.g. probability-weighted semantic entropy, Semantic Energy)
    to coexist with the baseline fields without forcing a schema migration.
    Known optional score fields are validated when present.
    """

    if not isinstance(record, Mapping):
        raise SchemaError("record must be a mapping")

    missing = sorted(REQUIRED_FIELDS - set(record))
    if missing:
        raise SchemaError(f"record is missing fields: {', '.join(missing)}")

    for field in (
        "run_id",
        "prompt_id",
        "dataset",
        "split",
        "prompt",
        "model",
        "entailment_backend",
    ):
        _require_string(record, field)

    _require_choice(record, "phase", PHASES)
    _require_choice(record, "cluster", CLUSTERS)
    _require_choice(record, "model_tier", MODEL_TIERS)

    reference_answers = _require_string_list(record, "reference_answers")
    sampled_answers = _require_string_list(record, "sampled_answers")
    normalized_answers = _require_string_list(record, "normalized_answers")
    semantic_clusters = _require_int_list(record, "semantic_clusters")
    cluster_representatives = _require_string_list(record, "cluster_representatives")

    if not reference_answers:
        raise SchemaError("reference_answers must not be empty")

    decoding = _require_mapping(record, "decoding")
    num_samples = _require_positive_int(decoding, "num_samples", "decoding")
    _require_finite_number(decoding, "temperature", "decoding")
    _require_finite_number(decoding, "top_p", "decoding")
    _require_positive_int(decoding, "max_new_tokens", "decoding")
    _require_int(decoding, "seed", "decoding")

    if len(sampled_answers) != num_samples:
        raise SchemaError("sampled_answers length must equal decoding.num_samples")
    if len(normalized_answers) != num_samples:
        raise SchemaError("normalized_answers length must equal decoding.num_samples")
    if len(semantic_clusters) != num_samples:
        raise SchemaError("semantic_clusters length must equal decoding.num_samples")
    if len(cluster_representatives) != len(set(semantic_clusters)):
        raise SchemaError(
            "cluster_representatives length must equal the number of observed clusters"
        )

    scores = _require_mapping(record, "scores")
    missing_scores = sorted(SCORE_FIELDS - set(scores))
    if missing_scores:
        raise SchemaError(f"scores is missing fields: {', '.join(missing_scores)}")
    score_fields_to_validate = SCORE_FIELDS | (OPTIONAL_SCORE_FIELDS & set(scores))
    for field in score_fields_to_validate:
        score = _require_finite_number(scores, field, "scores")
        if score < 0:
            raise SchemaError(f"scores.{field} must be non-negative")

    if not isinstance(record["correctness_label"], bool):
        raise SchemaError("correctness_label must be a boolean")


def _require_mapping(record: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = _get(record, field)
    if not isinstance(value, Mapping):
        raise SchemaError(f"{field} must be a mapping")
    return value


def _require_string(record: Mapping[str, Any], field: str) -> str:
    value = _get(record, field)
    if not isinstance(value, str):
        raise SchemaError(f"{field} must be a string")
    return value


def _require_choice(record: Mapping[str, Any], field: str, choices: set[str]) -> str:
    value = _require_string(record, field)
    if value not in choices:
        expected = ", ".join(sorted(choices))
        raise SchemaError(f"{field} must be one of: {expected}")
    return value


def _require_string_list(record: Mapping[str, Any], field: str) -> list[str]:
    value = _get(record, field)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SchemaError(f"{field} must be a list of strings")
    return value


def _require_int_list(record: Mapping[str, Any], field: str) -> list[int]:
    value = _get(record, field)
    if not isinstance(value, list) or not all(_is_int(item) for item in value):
        raise SchemaError(f"{field} must be a list of integers")
    return value


def _require_int(record: Mapping[str, Any], field: str, prefix: str) -> int:
    value = _get(record, field, prefix)
    if not _is_int(value):
        raise SchemaError(f"{prefix}.{field} must be an integer")
    return value


def _require_positive_int(record: Mapping[str, Any], field: str, prefix: str) -> int:
    value = _require_int(record, field, prefix)
    if value <= 0:
        raise SchemaError(f"{prefix}.{field} must be positive")
    return value


def _require_finite_number(
    record: Mapping[str, Any], field: str, prefix: str
) -> float:
    value = _get(record, field, prefix)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SchemaError(f"{prefix}.{field} must be a number")
    if not math.isfinite(value):
        raise SchemaError(f"{prefix}.{field} must be finite")
    return float(value)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _get(record: Mapping[str, Any], field: str, prefix: str | None = None) -> Any:
    if field not in record:
        name = f"{prefix}.{field}" if prefix else field
        raise SchemaError(f"{name} is required")
    return record[field]
