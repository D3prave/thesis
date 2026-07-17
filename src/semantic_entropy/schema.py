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
import re
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
CLUSTERS = {"local", "tinygpu", "alex"}
MODEL_TIERS = {"small", "7b_8b", "70b_plus"}
SCORE_FIELDS = {"surface_entropy", "discrete_semantic_entropy"}
# Optional score fields produced by Phase 2 extensions.  These are
# validated when present but their absence is not an error.
OPTIONAL_SCORE_FIELDS = {
    "naive_sample_entropy",
    # Probability-weighted estimators (scoring.py), require sequence_logprobs.
    # naive_entropy = paper's predictive-entropy baseline; semantic_entropy_full
    # = paper's primary probability-weighted semantic entropy. Both are entropies
    # in nats (finite, non-negative), so the standard score-field rule applies.
    "naive_entropy",
    "semantic_entropy_full",
    # Kernel Language Entropy (kle.py) — Nikitin et al. (2024).
    "kle",
    # Semantic Entropy Probe (probes.py) score — Kossen et al. (2024).
    # Stored as a probability of "uncertain" in [0, 1]; finite and
    # non-negative, so the same validation rule applies.
    "probe_uncertainty",
    # Separately named correctness-trained accuracy probe. Never aggregate or
    # report this field as a Semantic Entropy Probe.
    "accuracy_probe_uncertainty",
    # Long-form correctness confidence from the LLM judge (§5 of the bio
    # study plan).  Optional float in [0, 1]; the binary correctness_label
    # remains the AUROC label.
    "correctness_score",
    # P(True) supervised baseline (ptrue.py) — Kadavath et al. (2022).
    # Stored as uncertainty = 1 - P(True) in [0, 1].
    "ptrue_uncertainty",
    # Embedding-regression supervised baseline (train_embedding_regression.py)
    # — logistic probe on hidden states. Stored as p(incorrect) in [0, 1].
    "embedding_regression",
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
    if "top_k" in decoding:
        _require_int(decoding, "top_k", "decoding")
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

    # Legacy Phase 2 extension: per-sample states used by accuracy probes.
    # When present, must be a 2D list of floats with one row per sampled
    # answer and a consistent embedding dimension. Absence is the default.
    if "hidden_states" in record:
        _validate_hidden_states(record["hidden_states"], num_samples)

    # Genuine SEP feature: one greedy-response hidden state, kept separate
    # from the high-temperature response samples used to calculate DSE.
    if "sep_hidden_state" in record:
        _validate_sep_feature(record, decoding)

    # Optional: per-sequence length-normalized log-probabilities, captured at
    # generation time. Consumed by the probability-weighted estimators
    # (naive_entropy, semantic_entropy_full). One finite value per sample.
    if "sequence_logprobs" in record:
        _validate_sequence_logprobs(record["sequence_logprobs"], num_samples)

    # Optional: the single most-likely (low-temperature) answer used for the
    # paper-faithful accuracy rule. A free-text string when present.
    if "most_likely_answer" in record and not isinstance(record["most_likely_answer"], str):
        raise SchemaError("most_likely_answer must be a string")


def _validate_sequence_logprobs(value: Any, num_samples: int) -> None:
    if not isinstance(value, list):
        raise SchemaError("sequence_logprobs must be a list of floats")
    if len(value) != num_samples:
        raise SchemaError("sequence_logprobs length must equal decoding.num_samples")
    for index, entry in enumerate(value):
        if not isinstance(entry, (int, float)) or isinstance(entry, bool):
            raise SchemaError(f"sequence_logprobs[{index}] must be a number")
        if not math.isfinite(entry):
            raise SchemaError(f"sequence_logprobs[{index}] must be finite")


def _validate_hidden_states(value: Any, num_samples: int) -> None:
    if not isinstance(value, list):
        raise SchemaError("hidden_states must be a list of lists of floats")
    if len(value) != num_samples:
        raise SchemaError(
            "hidden_states length must equal decoding.num_samples"
        )
    if not value:
        return
    expected_dim: int | None = None
    for row_index, row in enumerate(value):
        if not isinstance(row, list):
            raise SchemaError(
                f"hidden_states[{row_index}] must be a list of floats"
            )
        if expected_dim is None:
            expected_dim = len(row)
        elif len(row) != expected_dim:
            raise SchemaError(
                f"hidden_states[{row_index}] has length {len(row)}; "
                f"expected {expected_dim} (must be rectangular)"
            )
        for col_index, entry in enumerate(row):
            if not isinstance(entry, (int, float)) or isinstance(entry, bool):
                raise SchemaError(
                    f"hidden_states[{row_index}][{col_index}] must be a number"
                )
            if not math.isfinite(entry):
                raise SchemaError(
                    f"hidden_states[{row_index}][{col_index}] must be finite"
                )


def _validate_sep_feature(
    record: Mapping[str, Any], decoding: Mapping[str, Any]
) -> None:
    value = record["sep_hidden_state"]
    if not isinstance(value, list) or not value:
        raise SchemaError("sep_hidden_state must be a non-empty list of floats")
    for index, entry in enumerate(value):
        if not isinstance(entry, (int, float)) or isinstance(entry, bool):
            raise SchemaError(f"sep_hidden_state[{index}] must be a number")
        if not math.isfinite(entry):
            raise SchemaError(f"sep_hidden_state[{index}] must be finite")

    if not isinstance(record.get("sep_greedy_answer"), str):
        raise SchemaError("sep_greedy_answer must be a string")
    metadata = record.get("sep_feature_metadata")
    if not isinstance(metadata, Mapping):
        raise SchemaError("sep_feature_metadata must be a mapping")
    required_metadata = {
        "protocol": "kossen_sep_2024",
        "feature_response_decoding": "greedy",
        "token_selection": "final_content_token_before_eos_or_eot",
        "layer_selection": "preregistered_final_layer",
        "generation_termination": "eos_or_eot",
        "generation_config_source": "model_generation_config_clone",
        "sampling_temperature": 1.0,
        "sampling_top_p": 0.9,
        "sampling_top_k": 50,
        "sampling_num_responses": 10,
    }
    for field, expected in required_metadata.items():
        if metadata.get(field) != expected:
            raise SchemaError(
                f"sep_feature_metadata.{field} must equal {expected!r}"
            )
    termination_token_ids = metadata.get("termination_token_ids")
    if (
        not isinstance(termination_token_ids, list)
        or not termination_token_ids
        or any(
            not isinstance(token_id, int) or isinstance(token_id, bool)
            for token_id in termination_token_ids
        )
        or termination_token_ids != sorted(set(termination_token_ids))
    ):
        raise SchemaError(
            "sep_feature_metadata.termination_token_ids must be a non-empty "
            "sorted list of unique integers"
        )
    for field in ("model_id", "tokenizer_id"):
        value = metadata.get(field)
        if not isinstance(value, str) or not value.strip():
            raise SchemaError(f"sep_feature_metadata.{field} must be non-empty")
    for field in ("model_revision", "tokenizer_revision"):
        value = metadata.get(field)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", value):
            raise SchemaError(
                f"sep_feature_metadata.{field} must be a full 40-character "
                "Hugging Face commit SHA"
            )
    if not isinstance(metadata.get("hidden_layer"), int) or isinstance(
        metadata.get("hidden_layer"), bool
    ):
        raise SchemaError("sep_feature_metadata.hidden_layer must be an integer")
    hidden_dimension = metadata.get("hidden_dimension")
    if (
        not isinstance(hidden_dimension, int)
        or isinstance(hidden_dimension, bool)
        or hidden_dimension != len(record["sep_hidden_state"])
    ):
        raise SchemaError(
            "sep_feature_metadata.hidden_dimension must equal the length of "
            "sep_hidden_state"
        )
    for field in (
        "generated_content_token_index",
        "full_sequence_token_index",
        "terminating_special_token_id",
        "sampled_eos_or_eot_count",
        "sampled_max_new_tokens_count",
        "base_seed",
        "prompt_seed",
    ):
        if not isinstance(metadata.get(field), int) or isinstance(
            metadata.get(field), bool
        ):
            raise SchemaError(f"sep_feature_metadata.{field} must be an integer")
    if metadata["terminating_special_token_id"] not in termination_token_ids:
        raise SchemaError(
            "sep_feature_metadata.terminating_special_token_id must be in "
            "termination_token_ids"
        )
    if (
        metadata["sampled_eos_or_eot_count"] < 0
        or metadata["sampled_max_new_tokens_count"] < 0
        or metadata["sampled_eos_or_eot_count"]
        + metadata["sampled_max_new_tokens_count"]
        != metadata["sampling_num_responses"]
    ):
        raise SchemaError(
            "SEP sampled termination counts must be non-negative and sum to "
            "sampling_num_responses"
        )

    expected_decoding = {
        "temperature": 1.0,
        "top_p": 0.9,
        "top_k": 50,
        "num_samples": 10,
        "seed": metadata["base_seed"],
    }
    for field, expected in expected_decoding.items():
        if decoding.get(field) != expected:
            raise SchemaError(
                f"decoding.{field} must equal SEP metadata value {expected!r}"
            )


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
