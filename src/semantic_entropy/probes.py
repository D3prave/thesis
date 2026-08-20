"""Semantic Entropy Probes (SEP): linear probes over hidden states.

SEP trains a small linear classifier on a generation model's *internal*
hidden states to predict a binarization of semantic entropy. The entropy
target comes from separate high-temperature samples, while the feature is
the final content token of one greedy response. A probe trained directly on
correctness is retained as a distinct accuracy-probe baseline; it is not a
Semantic Entropy Probe. Reference: Kossen et al. (2024), "Semantic Entropy
Probes".

The motivation is test-time cost: discrete semantic entropy requires
sampling ``M = 10`` generations per prompt plus pairwise entailment, while
SEP requires only a single forward pass per prompt.  When the probe
generalizes, it gives a cheap stand-in for the full entropy pipeline.

This module provides two top-level functions and a serialization-friendly
dataclass:

* :func:`train_probe` — fit an L2-regularized logistic regression from a
  set of scored records that include per-sample hidden states.
* :func:`score_probe` — return ``p(uncertain)`` in ``[0, 1]`` for a fresh
  hidden state vector.
* :class:`SEPProbe` — dataclass holding the trained coefficients and
  enough metadata to reconstruct the probe from JSON.

The training routine accepts records produced by the run-pipeline when
invoked with a ``ModelFnWithStates`` adapter (see
:mod:`semantic_entropy.models`). Canonical probes consume one
``sep_hidden_state`` vector from a separate greedy response; legacy accuracy
baselines may still consume a ``hidden_states`` matrix. Probe labels can be
``correctness_label`` (already a per-record bool) or
the *binarized* version of an existing entropy score. Canonical SEP-v2 derives
the entropy threshold from the training records with Eq. (5); artifacts made
with a caller-supplied threshold remain noncanonical and cannot be scored as SEP.

``numpy`` and ``scikit-learn`` are deferred imports so the rest of the
package keeps its stdlib-only runtime contract.  Installing the optional
extras (``pip install -e '.[sep]'``) brings both in.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from semantic_entropy.models import CANONICAL_SEP_FEATURE_MAX_NEW_TOKENS

if TYPE_CHECKING:
    import numpy as np


CANONICAL_SEP_ARTIFACT_SCHEMA = "semantic_entropy_probe_v3"
CANONICAL_SEP_TOKEN_SELECTION = "final_content_token_before_eos_or_eot"
CANONICAL_SEP_LAYER_SELECTION = "preregistered_final_layer"
CANONICAL_SEP_FEATURE_DEFINITION = (
    "greedy_response_final_content_token_before_eos_or_eot"
)
CANONICAL_SEP_TARGET_NLI_MODEL = "cross-encoder/nli-deberta-v3-large"
CANONICAL_SEP_TARGET_PROVENANCE_SCHEMA = "qconditioned_entropy_target_v1"
CANONICAL_SEP_MAX_SAMPLED_TRUNCATION_RATE = 0.01
_FULL_REVISION_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")

_SEP_FEATURE_IDENTITY_FIELDS = (
    "protocol",
    "feature_response_decoding",
    "token_selection",
    "layer_selection",
    "hidden_layer",
    "model_id",
    "model_revision",
    "tokenizer_id",
    "tokenizer_revision",
    "hidden_dimension",
    "generation_termination",
    "termination_token_ids",
    "generation_config_source",
    "feature_response_max_new_tokens",
)

_ENTROPY_TARGET_SHARED_FIELDS = (
    "question_conditioned",
    "question_conditioned_entailment",
    "entailment_input_format",
    "clustering_rule",
    "nli_model_id",
    "nli_model_revision",
    "nli_tokenizer_id",
    "nli_tokenizer_revision",
    "analysis_code_commit",
)

_SEP_CELL_DECODING_FIELDS = (
    "num_samples",
    "temperature",
    "top_p",
    "top_k",
    "max_new_tokens",
    "seed",
)

# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SEPProbe:
    """Serializable container for a trained SEP linear probe.

    The probe is a single-layer logistic regression
    ``p(uncertain | h) = sigmoid(coef . h + intercept)`` over a hidden
    state vector ``h`` of dimensionality ``hidden_dim``.

    Attributes:
        coef: One coefficient per input dimension.  Length must equal
            ``hidden_dim``.
        intercept: Scalar bias term.
        hidden_dim: Expected dimensionality of the input hidden state.
        label_source: ``"semantic_entropy_threshold"`` for a genuine SEP,
            or ``"correctness"`` for the separately named accuracy probe.
        probe_kind: Explicit artifact identity. Legacy entropy-target artifacts
            without this field are deliberately not upgraded to SEP.
        label_threshold: For ``label_source == "semantic_entropy_threshold"``,
            the threshold that was used to binarize the source entropy.
            ``None`` otherwise.
        source_score_field: For ``label_source == "semantic_entropy_threshold"``,
            the name of the entropy score that was binarized.  ``None``
            otherwise.
        train_size: Number of (sample, label) pairs used to fit the probe.
        positive_rate: Empirical fraction of *uncertain* labels in the
            training set.  Useful for sanity-checking class balance.
        pooling: Hidden-state aggregation strategy used during both training
            and inference.

            * ``"last"`` (default) — use the last generated token's hidden
              state. Matches the original SEP paper and is appropriate for
              short-answer QA.
            * ``"mean"`` — mean over all generated token positions. Recommended
              for long-form bio responses where the last token (often a period
              or EOS) is a noisy summary of the whole generation.
            * ``"mean_last_k"`` — mean over the last *k* generated token
              positions. Useful as a sensitivity ablation; *k* is stored in
              ``metadata["mean_last_k_k"]``.
        mean_last_k: Number of final tokens to average when
            ``pooling == "mean_last_k"``. Stored in ``metadata`` at save time
            and also as a standalone field for convenience.
    """

    coef: list[float]
    intercept: float
    hidden_dim: int
    label_source: Literal["correctness", "semantic_entropy_threshold"]
    probe_kind: Literal["semantic_entropy_probe", "accuracy_probe"] | None = None
    label_threshold: float | None = None
    source_score_field: str | None = None
    train_size: int = 0
    positive_rate: float = 0.0
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)
    pooling: Literal["last", "mean", "mean_last_k"] = "last"
    mean_last_k: int = 8  # used only when pooling == "mean_last_k"
    artifact_schema: str | None = CANONICAL_SEP_ARTIFACT_SCHEMA

    @property
    def effective_probe_kind(
        self,
    ) -> Literal["semantic_entropy_probe", "accuracy_probe"]:
        """Return the explicit kind without upgrading a legacy SEP artifact."""
        inferred = (
            "semantic_entropy_probe"
            if self.label_source == "semantic_entropy_threshold"
            else "accuracy_probe"
        )
        if self.probe_kind is None and inferred == "semantic_entropy_probe":
            raise ValueError(
                "semantic-entropy-target artifacts must explicitly declare "
                "probe_kind='semantic_entropy_probe'; legacy label-only artifacts "
                "are not canonical SEP artifacts"
            )
        if self.probe_kind is not None and self.probe_kind != inferred:
            raise ValueError(
                f"probe_kind={self.probe_kind!r} conflicts with "
                f"label_source={self.label_source!r}"
            )
        return self.probe_kind or inferred

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the probe."""
        return {
            "artifact_schema": self.artifact_schema,
            "probe_kind": self.effective_probe_kind,
            "coef": list(self.coef),
            "intercept": float(self.intercept),
            "hidden_dim": int(self.hidden_dim),
            "label_source": self.label_source,
            "label_threshold": (
                None if self.label_threshold is None else float(self.label_threshold)
            ),
            "source_score_field": self.source_score_field,
            "train_size": int(self.train_size),
            "positive_rate": float(self.positive_rate),
            "metadata": dict(self.metadata),
            "pooling": self.pooling,
            "mean_last_k": int(self.mean_last_k),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SEPProbe":
        """Reconstruct a probe from :meth:`to_dict` output."""
        label_source = payload["label_source"]
        if label_source not in {"correctness", "semantic_entropy_threshold"}:
            raise ValueError(f"unknown probe label_source={label_source!r}")
        probe = cls(
            coef=[float(c) for c in payload["coef"]],
            intercept=float(payload["intercept"]),
            hidden_dim=int(payload["hidden_dim"]),
            label_source=label_source,
            probe_kind=payload.get("probe_kind"),
            label_threshold=(
                None
                if payload.get("label_threshold") is None
                else float(payload["label_threshold"])
            ),
            source_score_field=payload.get("source_score_field"),
            train_size=int(payload.get("train_size", 0)),
            positive_rate=float(payload.get("positive_rate", 0.0)),
            metadata=dict(payload.get("metadata", {})),
            pooling=payload.get("pooling", "last"),
            mean_last_k=int(payload.get("mean_last_k", 8)),
            artifact_schema=payload.get("artifact_schema"),
        )
        probe.effective_probe_kind
        return probe

    def save(self, path: Path | str) -> None:
        """Write the probe to *path* as a single JSON file."""
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path | str) -> "SEPProbe":
        """Load a probe previously written by :meth:`save`."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Training and inference
# ---------------------------------------------------------------------------


def best_split_threshold(values: Sequence[float]) -> float:
    """Return the deterministic Eq. (5) semantic-entropy split threshold.

    Candidate thresholds are midpoints between adjacent distinct training
    values. The selected threshold minimizes the sum of within-group squared
    deviations for values below versus at-or-above the threshold. Ties choose
    the lower threshold. Evaluation values must never be passed here.
    """
    if len(values) < 2:
        raise ValueError("best-split threshold requires at least two values")
    cleaned: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"entropy value {index} must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"entropy value {index} must be finite")
        cleaned.append(numeric)

    unique = sorted(set(cleaned))
    if len(unique) < 2:
        raise ValueError(
            "best-split threshold requires at least two distinct entropy values"
        )

    candidates = [
        (left + right) / 2.0 for left, right in zip(unique, unique[1:])
    ]

    def objective(threshold: float) -> float:
        low = [value for value in cleaned if value < threshold]
        high = [value for value in cleaned if value >= threshold]
        low_mean = sum(low) / len(low)
        high_mean = sum(high) / len(high)
        return sum((value - low_mean) ** 2 for value in low) + sum(
            (value - high_mean) ** 2 for value in high
        )

    return min(candidates, key=lambda threshold: (objective(threshold), threshold))


def prompt_content_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    """Hash the exact held-out prompt set independently of model outputs."""

    rows: list[list[Any]] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        prompt_id = record.get("prompt_id")
        prompt = record.get("prompt")
        dataset = record.get("dataset")
        split = record.get("split")
        references = record.get("reference_answers")
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise ValueError(f"record {index}: prompt_id must be non-empty")
        if prompt_id in seen_ids:
            raise ValueError(f"records contain duplicate prompt_id={prompt_id!r}")
        seen_ids.add(prompt_id)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"record {index}: prompt must be non-empty")
        if not isinstance(dataset, str) or not dataset.strip():
            raise ValueError(f"record {index}: dataset must be non-empty")
        if not isinstance(split, str) or not split.strip():
            raise ValueError(f"record {index}: split must be non-empty")
        if not isinstance(references, list) or not all(
            isinstance(value, str) for value in references
        ):
            raise ValueError(f"record {index}: reference_answers must be strings")
        rows.append([prompt_id, dataset, split, prompt, references])
    payload = json.dumps(
        sorted(rows, key=lambda row: row[0]),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _entropy_target_record_provenance(
    record: Mapping[str, Any], *, role: str
) -> dict[str, Any]:
    """Validate and normalize one q-conditioned entropy-target lineage."""

    required_values = {
        "question_conditioned": True,
        "question_conditioned_entailment": True,
        "entailment_input_format": "question_answer_v1",
        "clustering_rule": "bidirectional",
        "nli_model_id": CANONICAL_SEP_TARGET_NLI_MODEL,
        "nli_tokenizer_id": CANONICAL_SEP_TARGET_NLI_MODEL,
    }
    for field_name, expected in required_values.items():
        if record.get(field_name) != expected:
            raise ValueError(
                f"{role}: entropy-target {field_name} must equal {expected!r}; "
                f"got {record.get(field_name)!r}"
            )

    normalized = dict(required_values)
    for field_name in (
        "nli_model_revision",
        "nli_tokenizer_revision",
        "analysis_code_commit",
    ):
        value = record.get(field_name)
        if not isinstance(value, str) or _FULL_REVISION_RE.fullmatch(value) is None:
            raise ValueError(
                f"{role}: entropy-target {field_name} must be a full 40-character "
                "commit SHA"
            )
        normalized[field_name] = value.lower()

    source_hash = record.get("source_artifact_sha256")
    if not isinstance(source_hash, str) or _SHA256_RE.fullmatch(source_hash) is None:
        raise ValueError(
            f"{role}: entropy-target source_artifact_sha256 must be SHA-256"
        )
    normalized["source_artifact_sha256"] = source_hash.lower()
    return normalized


def _common_entropy_target_provenance(
    records: Sequence[Mapping[str, Any]], *, role: str
) -> dict[str, Any]:
    if not records:
        raise ValueError(f"{role}: entropy-target records must not be empty")
    first = _entropy_target_record_provenance(records[0], role=f"{role} record 0")
    for index, record in enumerate(records[1:], start=1):
        current = _entropy_target_record_provenance(
            record, role=f"{role} record {index}"
        )
        if current != first:
            differing = sorted(
                field
                for field in first
                if current.get(field) != first.get(field)
            )
            raise ValueError(
                f"{role} entropy-target provenance differs across records: "
                f"{', '.join(differing)}"
            )
    return first


def validate_entropy_target_provenance(
    value: Mapping[str, Any], *, role: str = "entropy_target_provenance"
) -> dict[str, Any]:
    """Validate the serialized train/eval q-conditioned target lineage."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{role} must be a mapping")
    expected_values = {
        "schema": CANONICAL_SEP_TARGET_PROVENANCE_SCHEMA,
        "question_conditioned": True,
        "question_conditioned_entailment": True,
        "entailment_input_format": "question_answer_v1",
        "clustering_rule": "bidirectional",
        "nli_model_id": CANONICAL_SEP_TARGET_NLI_MODEL,
        "nli_tokenizer_id": CANONICAL_SEP_TARGET_NLI_MODEL,
    }
    normalized = dict(expected_values)
    for field_name, expected in expected_values.items():
        if value.get(field_name) != expected:
            raise ValueError(
                f"{role}.{field_name} must equal {expected!r}; "
                f"got {value.get(field_name)!r}"
            )
    for field_name in (
        "nli_model_revision",
        "nli_tokenizer_revision",
        "analysis_code_commit",
    ):
        field_value = value.get(field_name)
        if (
            not isinstance(field_value, str)
            or _FULL_REVISION_RE.fullmatch(field_value) is None
        ):
            raise ValueError(
                f"{role}.{field_name} must be a full 40-character commit SHA"
            )
        normalized[field_name] = field_value.lower()
    for field_name in (
        "train_source_artifact_sha256",
        "eval_source_artifact_sha256",
    ):
        field_value = value.get(field_name)
        if not isinstance(field_value, str) or _SHA256_RE.fullmatch(field_value) is None:
            raise ValueError(f"{role}.{field_name} must be SHA-256")
        normalized[field_name] = field_value.lower()
    return normalized


def canonical_entropy_target_provenance(
    train_records: Sequence[Mapping[str, Any]],
    eval_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind a canonical SEP target to one identical q-conditioned method."""

    train = _common_entropy_target_provenance(train_records, role="training")
    evaluation = _common_entropy_target_provenance(
        eval_records, role="evaluation"
    )
    for field_name in _ENTROPY_TARGET_SHARED_FIELDS:
        if train[field_name] != evaluation[field_name]:
            raise ValueError(
                "training/evaluation entropy-target provenance differs for "
                f"{field_name}: {train[field_name]!r} != "
                f"{evaluation[field_name]!r}"
            )
    return validate_entropy_target_provenance(
        {
            "schema": CANONICAL_SEP_TARGET_PROVENANCE_SCHEMA,
            **{
                field_name: train[field_name]
                for field_name in _ENTROPY_TARGET_SHARED_FIELDS
            },
            "train_source_artifact_sha256": train["source_artifact_sha256"],
            "eval_source_artifact_sha256": evaluation["source_artifact_sha256"],
        }
    )


def validate_sep_cell_identity(
    value: Mapping[str, Any], *, role: str = "SEP cell identity"
) -> dict[str, Any]:
    """Validate one serialized dataset/model/decoding identity for SEP."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{role} must be a mapping")
    normalized: dict[str, Any] = {}
    for field_name in ("dataset", "model_id"):
        field_value = value.get(field_name)
        if not isinstance(field_value, str) or not field_value.strip():
            raise ValueError(f"{role}.{field_name} must be non-empty")
        normalized[field_name] = field_value

    decoding = value.get("decoding")
    if not isinstance(decoding, Mapping):
        raise ValueError(f"{role}.decoding must be a mapping")
    normalized_decoding: dict[str, Any] = {}
    canonical_integers = {"num_samples": 10, "top_k": 50}
    for field, expected in canonical_integers.items():
        field_value = decoding.get(field)
        if (
            isinstance(field_value, bool)
            or not isinstance(field_value, int)
            or field_value != expected
        ):
            raise ValueError(
                f"{role}.decoding.{field} must equal {expected!r}; "
                f"got {field_value!r}"
            )
        normalized_decoding[field] = field_value
    canonical_floats = {"temperature": 1.0, "top_p": 0.9}
    for field, expected in canonical_floats.items():
        field_value = decoding.get(field)
        if (
            isinstance(field_value, bool)
            or not isinstance(field_value, (int, float))
            or not math.isfinite(float(field_value))
            or float(field_value) != expected
        ):
            raise ValueError(
                f"{role}.decoding.{field} must equal {expected!r}; "
                f"got {field_value!r}"
            )
        normalized_decoding[field] = float(field_value)
    max_new_tokens = decoding.get("max_new_tokens")
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens <= 0
    ):
        raise ValueError(
            f"{role}.decoding.max_new_tokens must be a positive integer"
        )
    normalized_decoding["max_new_tokens"] = max_new_tokens
    seed = decoding.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"{role}.decoding.seed must be an integer")
    normalized_decoding["seed"] = seed
    normalized["decoding"] = {
        field: normalized_decoding[field] for field in _SEP_CELL_DECODING_FIELDS
    }
    return normalized


def _record_sep_cell_identity(
    record: Mapping[str, Any], *, role: str
) -> dict[str, Any]:
    identity = validate_sep_cell_identity(
        {
            "dataset": record.get("dataset"),
            "model_id": record.get("model"),
            "decoding": record.get("decoding"),
        },
        role=role,
    )
    feature_metadata = record.get("sep_feature_metadata")
    if not isinstance(feature_metadata, Mapping):
        raise ValueError(f"{role}.sep_feature_metadata must be a mapping")
    if feature_metadata.get("model_id") != identity["model_id"]:
        raise ValueError(
            f"{role}.sep_feature_metadata.model_id must equal record.model"
        )
    metadata_fields = {
        "sampling_num_responses": ("num_samples", int),
        "sampling_temperature": ("temperature", float),
        "sampling_top_p": ("top_p", float),
        "sampling_top_k": ("top_k", int),
        "sampling_max_new_tokens": ("max_new_tokens", int),
        "base_seed": ("seed", int),
    }
    decoding = identity["decoding"]
    for metadata_field, (decoding_field, expected_type) in metadata_fields.items():
        actual = feature_metadata.get(metadata_field)
        expected = decoding[decoding_field]
        valid_type = (
            isinstance(actual, (int, float))
            if expected_type is float
            else isinstance(actual, int)
        ) and not isinstance(actual, bool)
        if (
            not valid_type
            or (expected_type is float and not math.isfinite(float(actual)))
            or float(actual) != float(expected)
        ):
            raise ValueError(
                f"{role}.sep_feature_metadata.{metadata_field} must equal "
                f"decoding.{decoding_field}={expected!r}; got {actual!r}"
            )
    return identity


def canonical_sep_cell_identity(
    records: Sequence[Mapping[str, Any]], *, role: str
) -> dict[str, Any]:
    """Validate that every record in one split belongs to one SEP cell."""

    if not records:
        raise ValueError(f"{role} SEP cell records must not be empty")
    first = _record_sep_cell_identity(records[0], role=f"{role} record 0")
    for index, record in enumerate(records[1:], start=1):
        current = _record_sep_cell_identity(record, role=f"{role} record {index}")
        if current != first:
            differing = [
                field
                for field in ("dataset", "model_id")
                if current[field] != first[field]
            ]
            differing.extend(
                f"decoding.{field}"
                for field in _SEP_CELL_DECODING_FIELDS
                if current["decoding"][field] != first["decoding"][field]
            )
            raise ValueError(
                f"{role} SEP cell identity differs across records: "
                f"{', '.join(differing)}"
            )
    return first


def canonical_shared_sep_cell_identity(
    train_records: Sequence[Mapping[str, Any]],
    eval_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Require training and evaluation records to share one exact SEP cell."""

    training = canonical_sep_cell_identity(train_records, role="training")
    evaluation = canonical_sep_cell_identity(eval_records, role="evaluation")
    if training != evaluation:
        differing = [
            field
            for field in ("dataset", "model_id")
            if training[field] != evaluation[field]
        ]
        differing.extend(
            f"decoding.{field}"
            for field in _SEP_CELL_DECODING_FIELDS
            if training["decoding"][field] != evaluation["decoding"][field]
        )
        raise ValueError(
            "training/evaluation SEP cell identity differs for "
            f"{', '.join(differing)}"
        )
    return training


def _sep_feature_identity(metadata: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    """Return and validate the immutable identity of one canonical SEP feature."""

    identity = {field: metadata.get(field) for field in _SEP_FEATURE_IDENTITY_FIELDS}
    required_values = {
        "protocol": "kossen_sep_2024",
        "feature_response_decoding": "greedy",
        "token_selection": CANONICAL_SEP_TOKEN_SELECTION,
        "layer_selection": CANONICAL_SEP_LAYER_SELECTION,
        "generation_termination": "eos_or_eot",
        "generation_config_source": "model_generation_config_clone",
        "feature_response_max_new_tokens": (
            CANONICAL_SEP_FEATURE_MAX_NEW_TOKENS
        ),
    }
    for field_name, expected in required_values.items():
        if identity[field_name] != expected:
            raise ValueError(
                f"{role}: SEP feature {field_name} must equal {expected!r}; "
                f"got {identity[field_name]!r}"
            )
    for field_name in ("model_id", "tokenizer_id"):
        if (
            not isinstance(identity[field_name], str)
            or not identity[field_name].strip()
        ):
            raise ValueError(f"{role}: SEP feature {field_name} must be non-empty")
    for field_name in ("model_revision", "tokenizer_revision"):
        if not isinstance(
            identity[field_name], str
        ) or not _FULL_REVISION_RE.fullmatch(
            identity[field_name]
        ):
            raise ValueError(
                f"{role}: SEP feature {field_name} must be a full 40-character "
                "Hugging Face commit SHA"
            )
    for field in ("hidden_layer", "hidden_dimension"):
        if not isinstance(identity[field], int) or isinstance(identity[field], bool):
            raise ValueError(f"{role}: SEP feature {field} must be an integer")
    if identity["hidden_dimension"] <= 0:
        raise ValueError(f"{role}: SEP feature hidden_dimension must be positive")
    termination_token_ids = identity["termination_token_ids"]
    if (
        not isinstance(termination_token_ids, list)
        or not termination_token_ids
        or any(
            not isinstance(token_id, int) or isinstance(token_id, bool)
            for token_id in termination_token_ids
        )
        or termination_token_ids != sorted(set(termination_token_ids))
    ):
        raise ValueError(
            f"{role}: SEP feature termination_token_ids must be a non-empty "
            "sorted list of unique integers"
        )
    return identity


def _validate_sampled_truncation_audit(metadata: Mapping[str, Any]) -> None:
    limit = metadata.get("sampled_truncation_rate_limit")
    if (
        isinstance(limit, bool)
        or not isinstance(limit, (int, float))
        or float(limit) != CANONICAL_SEP_MAX_SAMPLED_TRUNCATION_RATE
    ):
        raise ValueError(
            "canonical SEP metadata.sampled_truncation_rate_limit must equal "
            f"{CANONICAL_SEP_MAX_SAMPLED_TRUNCATION_RATE}"
        )
    for role in ("train", "eval"):
        total = metadata.get(f"{role}_sampled_response_count")
        truncated = metadata.get(f"{role}_sampled_max_new_tokens_count")
        rate = metadata.get(f"{role}_sampled_truncation_rate")
        if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
            raise ValueError(
                f"canonical SEP metadata.{role}_sampled_response_count must be "
                "positive"
            )
        if (
            isinstance(truncated, bool)
            or not isinstance(truncated, int)
            or not 0 <= truncated <= total
        ):
            raise ValueError(
                f"canonical SEP metadata.{role}_sampled_max_new_tokens_count "
                "must be between zero and the sampled response count"
            )
        expected_rate = truncated / total
        if (
            isinstance(rate, bool)
            or not isinstance(rate, (int, float))
            or not math.isfinite(float(rate))
            or not math.isclose(float(rate), expected_rate, rel_tol=0.0, abs_tol=1e-15)
        ):
            raise ValueError(
                f"canonical SEP metadata.{role}_sampled_truncation_rate must "
                "match the audited counts"
            )
        if truncated * 100 > total:
            raise ValueError(
                f"canonical SEP {role} sampled max-token truncation rate "
                f"{expected_rate:.6f} exceeds the 1% limit"
            )


def _validate_canonical_probe_structure_and_split(
    probe: SEPProbe,
    *,
    expected_kind: Literal["semantic_entropy_probe", "accuracy_probe"],
    expected_target: str,
    expected_threshold_selection: str,
    expected_threshold_fit_scope: str,
) -> None:
    """Validate fields shared by canonical SEP and accuracy probes."""

    if probe.effective_probe_kind != expected_kind:
        raise ValueError(
            f"artifact kind is {probe.effective_probe_kind!r}, expected "
            f"{expected_kind!r}"
        )
    if probe.artifact_schema != CANONICAL_SEP_ARTIFACT_SCHEMA:
        raise ValueError(
            "canonical probe scoring requires artifact_schema="
            f"{CANONICAL_SEP_ARTIFACT_SCHEMA!r}; got {probe.artifact_schema!r}"
        )
    if probe.pooling != "last":
        raise ValueError("canonical probe pooling must be 'last'")
    if probe.hidden_dim <= 0 or len(probe.coef) != probe.hidden_dim:
        raise ValueError("canonical probe coefficient dimension is invalid")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in probe.coef
    ) or not math.isfinite(float(probe.intercept)):
        raise ValueError("canonical probe coefficients and intercept must be finite")
    if probe.train_size <= 0:
        raise ValueError("canonical probe train_size must be positive")
    if not math.isfinite(probe.positive_rate) or not 0.0 <= probe.positive_rate <= 1.0:
        raise ValueError("canonical probe positive_rate must be in [0, 1]")

    metadata = probe.metadata
    required_metadata = {
        "artifact_schema": CANONICAL_SEP_ARTIFACT_SCHEMA,
        "target_definition": expected_target,
        "feature_definition": CANONICAL_SEP_FEATURE_DEFINITION,
        "threshold_selection": expected_threshold_selection,
        "threshold_fit_scope": expected_threshold_fit_scope,
        "probe_fit_scope": "training_records_only",
        "feature_aggregate": "first",
        "split_overlap_count": 0,
        "split_content_overlap_count": 0,
    }
    for field_name, expected in required_metadata.items():
        if metadata.get(field_name) != expected:
            raise ValueError(
                f"canonical probe metadata.{field_name} must equal {expected!r}; "
                f"got {metadata.get(field_name)!r}"
            )
    analysis_code_commit = metadata.get("analysis_code_commit")
    if (
        not isinstance(analysis_code_commit, str)
        or _FULL_REVISION_RE.fullmatch(analysis_code_commit) is None
    ):
        raise ValueError(
            "canonical probe metadata.analysis_code_commit must be a full "
            "40-character commit SHA"
        )
    inverse_l2 = metadata.get("inverse_l2_regularization")
    if (
        isinstance(inverse_l2, bool)
        or not isinstance(inverse_l2, (int, float))
        or not math.isfinite(float(inverse_l2))
        or float(inverse_l2) <= 0.0
    ):
        raise ValueError(
            "canonical probe metadata.inverse_l2_regularization must be positive"
        )
    training_seed = metadata.get("training_seed")
    if isinstance(training_seed, bool) or not isinstance(training_seed, int):
        raise ValueError("canonical probe metadata.training_seed must be an integer")
    for role in ("train", "eval"):
        count = metadata.get(f"{role}_prompt_count")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError(
                f"canonical SEP metadata.{role}_prompt_count must be positive"
            )
        for suffix in (
            "prompt_ids_sha256",
            "prompt_content_sha256",
            "prompt_identity_sha256",
        ):
            value = metadata.get(f"{role}_{suffix}")
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(
                    f"canonical SEP metadata.{role}_{suffix} must be a SHA-256"
                )
    # train_prompt_count describes the locked training split; the probe is fitted
    # only on the prompts whose greedy feature is present, so compare against the
    # fitted count when the artifact records one (older artifacts predate the
    # field and had no exclusions, so the two counts coincide).
    fitted_count = metadata.get("train_prompt_fitted_count")
    if fitted_count is None:
        fitted_count = metadata["train_prompt_count"]
    if isinstance(fitted_count, bool) or not isinstance(fitted_count, int):
        raise ValueError(
            "canonical SEP metadata.train_prompt_fitted_count must be an integer"
        )
    if fitted_count != probe.train_size:
        raise ValueError(
            "canonical SEP fitted train prompt count must equal probe.train_size"
        )
    feature_identity = metadata.get("feature_identity")
    if not isinstance(feature_identity, Mapping):
        raise ValueError("canonical SEP metadata.feature_identity must be a mapping")
    validated_identity = _sep_feature_identity(
        feature_identity, role="probe metadata.feature_identity"
    )
    if validated_identity["hidden_dimension"] != probe.hidden_dim:
        raise ValueError(
            "probe feature hidden_dimension does not match probe.hidden_dim"
        )
    cell_identity = validate_sep_cell_identity(
        metadata.get("cell_identity"),
        role="canonical probe metadata.cell_identity",
    )
    if cell_identity["model_id"] != validated_identity["model_id"]:
        raise ValueError(
            "canonical probe cell_identity.model_id must match "
            "feature_identity.model_id"
        )


def validate_canonical_sep_probe(probe: SEPProbe) -> None:
    """Reject any artifact that does not meet the canonical SEP-v2 contract."""

    if probe.label_source != "semantic_entropy_threshold":
        raise ValueError("canonical SEP must target thresholded semantic entropy")
    if (
        probe.label_threshold is None
        or isinstance(probe.label_threshold, bool)
        or not math.isfinite(float(probe.label_threshold))
    ):
        raise ValueError("canonical SEP label_threshold must be finite")
    if not isinstance(probe.source_score_field, str) or not probe.source_score_field:
        raise ValueError("canonical SEP source_score_field must be non-empty")
    _validate_canonical_probe_structure_and_split(
        probe,
        expected_kind="semantic_entropy_probe",
        expected_target="high_semantic_entropy",
        expected_threshold_selection="eq5_best_split_training_records_only",
        expected_threshold_fit_scope="training_records_only",
    )
    target_provenance = probe.metadata.get("entropy_target_provenance")
    validate_entropy_target_provenance(
        target_provenance,
        role="canonical SEP metadata.entropy_target_provenance",
    )
    _validate_sampled_truncation_audit(probe.metadata)


def validate_canonical_accuracy_probe(probe: SEPProbe) -> None:
    """Reject correctness probes that do not use the canonical held-out feature."""

    if probe.probe_kind != "accuracy_probe":
        raise ValueError(
            "canonical accuracy probe must explicitly declare "
            "probe_kind='accuracy_probe'"
        )
    if probe.label_source != "correctness":
        raise ValueError("canonical accuracy probe must target correctness")
    if probe.label_threshold is not None or probe.source_score_field is not None:
        raise ValueError(
            "canonical accuracy probe cannot carry an entropy threshold or source"
        )
    _validate_canonical_probe_structure_and_split(
        probe,
        expected_kind="accuracy_probe",
        expected_target="model_incorrectness",
        expected_threshold_selection="not_applicable",
        expected_threshold_fit_scope="not_applicable",
    )


def _validate_scoring_prompt_content(
    records: Sequence[Mapping[str, Any]], probe: SEPProbe
) -> None:
    expected_count = probe.metadata["eval_prompt_count"]
    if len(records) != expected_count:
        raise ValueError(
            f"held-out record count {len(records)} does not match the probe audit "
            f"count {expected_count}"
        )
    actual_hash = prompt_content_sha256(records)
    expected_hash = probe.metadata["eval_prompt_content_sha256"]
    if actual_hash != expected_hash:
        raise ValueError(
            "held-out prompt-content hash does not match the probe's audited "
            "evaluation split"
        )


def validate_canonical_sep_scoring_prompt_content(
    records: Sequence[Mapping[str, Any]], probe: SEPProbe
) -> None:
    """Bind a canonical SEP to the exact audited held-out prompt content."""

    validate_canonical_sep_probe(probe)
    _validate_scoring_prompt_content(records, probe)


def validate_canonical_accuracy_scoring_prompt_content(
    records: Sequence[Mapping[str, Any]], probe: SEPProbe
) -> None:
    """Bind an accuracy probe to the exact audited held-out prompt content."""

    validate_canonical_accuracy_probe(probe)
    _validate_scoring_prompt_content(records, probe)


def _validate_scoring_record_features(
    records: Sequence[Mapping[str, Any]], probe: SEPProbe
) -> None:
    expected_identity = dict(probe.metadata["feature_identity"])
    expected_cell_identity = validate_sep_cell_identity(
        probe.metadata["cell_identity"], role="probe metadata.cell_identity"
    )
    for index, record in enumerate(records):
        actual_cell_identity = _record_sep_cell_identity(
            record, role=f"evaluation record {index}"
        )
        if actual_cell_identity != expected_cell_identity:
            raise ValueError(
                f"evaluation record {index}: SEP cell identity does not match "
                "the trained probe"
            )
        feature_metadata = record.get("sep_feature_metadata")
        if not isinstance(feature_metadata, Mapping):
            raise ValueError(f"evaluation record {index}: sep_feature_metadata missing")
        actual_identity = _sep_feature_identity(
            feature_metadata, role=f"evaluation record {index}"
        )
        if actual_identity != expected_identity:
            raise ValueError(
                f"evaluation record {index}: SEP feature identity does not match "
                "the trained probe"
            )
        hidden = record.get("sep_hidden_state")
        if not isinstance(hidden, list) or len(hidden) != probe.hidden_dim:
            raise ValueError(
                f"evaluation record {index}: sep_hidden_state dimension does not "
                f"match probe.hidden_dim={probe.hidden_dim}"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in hidden
        ):
            raise ValueError(
                f"evaluation record {index}: sep_hidden_state must contain only "
                "finite numbers"
            )


def sep_feature_unavailable(record: Mapping[str, Any]) -> bool:
    """True when a record carries no usable probe feature.

    ``greedy_degenerate_excluded`` alone no longer answers this. Under few-shot
    raw completion greedy decoding almost never emits EOS, so the flag is set on
    most records; where a feature was later recovered the record declares
    ``feature_backfill`` and keeps a usable ``sep_hidden_state``. Only records
    that are degenerate *and* carry no backfill are genuinely featureless.
    """

    metadata = record.get("sep_feature_metadata")
    if not isinstance(metadata, Mapping):
        return False
    if metadata.get("greedy_degenerate_excluded") is not True:
        return False
    return metadata.get("feature_backfill") is None


def _partition_feature_present(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Split records into feature-present and genuinely feature-less."""

    present: list[Mapping[str, Any]] = []
    excluded: list[Mapping[str, Any]] = []
    for record in records:
        if sep_feature_unavailable(record):
            excluded.append(record)
        else:
            present.append(record)
    return present, excluded


def _validate_excluded_scoring_records(
    excluded: Sequence[Mapping[str, Any]], probe: SEPProbe
) -> None:
    """Validate the degenerate-greedy prompts kept in the locked split.

    These prompts carry no probe feature by construction, so the feature
    checks cannot apply to them. They must still belong to the probe's cell,
    and their absence of a feature must be the *declared* exclusion rather
    than a silently missing or malformed hidden state.
    """

    expected_cell_identity = validate_sep_cell_identity(
        probe.metadata["cell_identity"], role="probe metadata.cell_identity"
    )
    for index, record in enumerate(excluded):
        actual_cell_identity = _record_sep_cell_identity(
            record, role=f"excluded evaluation record {index}"
        )
        if actual_cell_identity != expected_cell_identity:
            raise ValueError(
                f"excluded evaluation record {index}: SEP cell identity does "
                "not match the trained probe"
            )
        if record.get("sep_hidden_state") is not None:
            raise ValueError(
                f"excluded evaluation record {index}: greedy_degenerate_"
                "excluded records must carry a null sep_hidden_state"
            )


def validate_canonical_sep_scoring_records(
    records: Sequence[Mapping[str, Any]], probe: SEPProbe
) -> None:
    """Bind a canonical SEP to exact held-out records and feature identity.

    ``records`` must be exactly the record set the probe audited, i.e. the
    feature-present subset of the locked evaluation split: the prompt count
    and content hash are bound to it (see ``train_sep_probe``'s
    ``split_audit_metadata``, and the sealed v2 artifacts, which record
    feature-present counts). Should a caller pass a set that still contains
    declared degenerate-greedy prompts, the feature checks are applied to the
    feature-present members and the excluded ones are separately required to
    be exactly the declared, feature-less shape — so a null hidden state can
    never pass as a feature, whichever set is supplied.
    """

    validate_canonical_sep_scoring_prompt_content(records, probe)
    present, excluded = _partition_feature_present(records)
    _validate_scoring_record_features(present, probe)
    _validate_excluded_scoring_records(excluded, probe)


def validate_canonical_accuracy_scoring_records(
    records: Sequence[Mapping[str, Any]], probe: SEPProbe
) -> None:
    """Bind an accuracy probe to exact held-out records and feature identity.

    Same split semantics as :func:`validate_canonical_sep_scoring_records`.
    """

    validate_canonical_accuracy_scoring_prompt_content(records, probe)
    present, excluded = _partition_feature_present(records)
    _validate_scoring_record_features(present, probe)
    _validate_excluded_scoring_records(excluded, probe)


def _canonical_probe_provenance(probe: SEPProbe) -> dict[str, Any]:
    metadata = probe.metadata
    artifact_payload = (
        json.dumps(probe.to_dict(), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    provenance = {
        "artifact_schema": probe.artifact_schema,
        "probe_kind": probe.effective_probe_kind,
        "probe_artifact_sha256": hashlib.sha256(artifact_payload).hexdigest(),
        "label_source": probe.label_source,
        "label_threshold": (
            None
            if probe.label_threshold is None
            else float(probe.label_threshold)
        ),
        "source_score_field": probe.source_score_field,
        "threshold_selection": metadata["threshold_selection"],
        "threshold_fit_scope": metadata["threshold_fit_scope"],
        "analysis_code_commit": metadata["analysis_code_commit"],
        "inverse_l2_regularization": metadata["inverse_l2_regularization"],
        "training_seed": metadata["training_seed"],
        "split_overlap_count": metadata["split_overlap_count"],
        "split_content_overlap_count": metadata["split_content_overlap_count"],
        "train_prompt_count": metadata["train_prompt_count"],
        "train_prompt_content_sha256": metadata["train_prompt_content_sha256"],
        "train_prompt_identity_sha256": metadata["train_prompt_identity_sha256"],
        "eval_prompt_count": metadata["eval_prompt_count"],
        "eval_prompt_content_sha256": metadata["eval_prompt_content_sha256"],
        "eval_prompt_identity_sha256": metadata["eval_prompt_identity_sha256"],
        "feature_definition": metadata["feature_definition"],
        "cell_identity": dict(metadata["cell_identity"]),
        "feature_identity": dict(metadata["feature_identity"]),
    }
    if probe.effective_probe_kind == "semantic_entropy_probe":
        provenance.update(
            {
                "entropy_target_provenance": dict(
                    metadata["entropy_target_provenance"]
                ),
                "sampled_truncation_rate_limit": metadata[
                    "sampled_truncation_rate_limit"
                ],
                "train_sampled_response_count": metadata[
                    "train_sampled_response_count"
                ],
                "train_sampled_max_new_tokens_count": metadata[
                    "train_sampled_max_new_tokens_count"
                ],
                "train_sampled_truncation_rate": metadata[
                    "train_sampled_truncation_rate"
                ],
                "eval_sampled_response_count": metadata[
                    "eval_sampled_response_count"
                ],
                "eval_sampled_max_new_tokens_count": metadata[
                    "eval_sampled_max_new_tokens_count"
                ],
                "eval_sampled_truncation_rate": metadata[
                    "eval_sampled_truncation_rate"
                ],
            }
        )
    return provenance


def canonical_sep_provenance(probe: SEPProbe) -> dict[str, Any]:
    """Return compact, coefficient-free provenance for a canonical SEP score."""

    validate_canonical_sep_probe(probe)
    return _canonical_probe_provenance(probe)


def canonical_accuracy_probe_provenance(probe: SEPProbe) -> dict[str, Any]:
    """Return compact provenance for a separately named accuracy probe score."""

    validate_canonical_accuracy_probe(probe)
    return _canonical_probe_provenance(probe)


def _pool_hidden_states(
    arr: "np.ndarray",  # shape (num_samples, hidden_dim)
    pooling: str,
    mean_last_k: int = 8,
) -> "np.ndarray | None":
    """Pool a 2-D hidden-state array into a 1-D feature vector.

    Args:
        arr: ``(num_samples, hidden_dim)`` array.
        pooling: One of ``"last"``, ``"mean"``, ``"mean_last_k"``.
        mean_last_k: Number of last samples to average for ``"mean_last_k"``.

    Returns:
        1-D feature vector, or ``None`` when *pooling* is ``"last"`` (so the
        caller can fall through to the legacy ``aggregate`` logic).
    """
    if pooling == "last":
        return None  # let caller handle via aggregate keyword
    if pooling == "mean":
        return arr.mean(axis=0)
    if pooling == "mean_last_k":
        k = max(1, min(mean_last_k, arr.shape[0]))
        return arr[-k:].mean(axis=0)
    raise ValueError(
        f"unknown pooling: {pooling!r}; expected 'last', 'mean', or 'mean_last_k'"
    )


def train_probe(
    records: Sequence[Mapping[str, Any]],
    *,
    label_source: Literal["correctness", "semantic_entropy_threshold"] = "correctness",
    source_score_field: str | None = None,
    label_threshold: float | None = None,
    aggregate: Literal["first", "mean"] = "first",
    pooling: Literal["last", "mean", "mean_last_k"] = "last",
    mean_last_k: int = 8,
    C: float = 1.0,
    seed: int = 0,
    metadata: Mapping[str, Any] | None = None,
) -> SEPProbe:
    """Fit a logistic-regression probe on per-record hidden states.

    Args:
        records: Iterable of scored records. Each record must contain a
            singular ``sep_hidden_state`` from a greedy response for SEP, or
            a legacy ``hidden_states`` matrix for an accuracy probe, plus the
            relevant target field.
        label_source: Defines what the probe predicts.

            * ``"correctness"`` — the positive class is *incorrect*
              (``not correctness_label``), matching the AUROC convention
              used throughout the metrics module. This creates an explicitly
              identified accuracy probe, not SEP.
            * ``"semantic_entropy_threshold"`` — the positive class is
              ``scores[source_score_field] >= label_threshold``. If no
              threshold is supplied, Eq. (5)'s best split is derived from
              these training records only. This creates a genuine SEP.

        source_score_field: Name of the entropy score under
            ``records[i]["scores"]`` to binarize.  Required when
            ``label_source == "semantic_entropy_threshold"``.
        label_threshold: Threshold for the entropy binarization. When omitted
            for SEP, derive it from the training records only. A supplied value
            creates a noncanonical artifact that SEP scoring rejects.
        aggregate: How to convert the per-sample hidden states of a single
            record into one feature vector for training.

            * ``"first"`` (default) — for an accuracy probe over legacy
              records, use the first sampled generation's state.
            * ``"mean"`` — average the per-sample hidden states.  This is
              useful when the probe is trained from records that already
              have ``M`` samples and we want to give the trainer the
              denoised mean.

        C: Inverse-regularization strength forwarded to
            ``sklearn.linear_model.LogisticRegression``.  Larger means
            weaker L2 regularization.
        seed: Random seed forwarded to the scikit-learn solver.
        metadata: Additional audited provenance to serialize with the probe.
            Canonical entropy-target training requires the complete
            ``entropy_target_provenance`` produced by the train/evaluation split
            audit.

    Returns:
        A trained :class:`SEPProbe`.

    Raises:
        ValueError: If *records* is empty, the hidden-state shapes are
            inconsistent, or the label distribution is degenerate (only
            one class observed).
        ImportError: If ``numpy`` or ``scikit-learn`` are not installed.
    """
    if not records:
        raise ValueError("records must not be empty")
    if (
        isinstance(C, bool)
        or not isinstance(C, (int, float))
        or not math.isfinite(float(C))
        or float(C) <= 0.0
    ):
        raise ValueError("C must be a positive finite number")

    threshold_was_derived = False
    target_provenance: dict[str, Any] | None = None
    if label_source == "semantic_entropy_threshold":
        if source_score_field is None:
            raise ValueError(
                "label_source='semantic_entropy_threshold' requires "
                "source_score_field to be set"
            )
        if pooling != "last" or aggregate != "first":
            raise ValueError(
                "semantic_entropy_threshold SEP uses one greedy final-content-token "
                "feature; aggregate must be 'first' and pooling must be 'last'"
            )
        if label_threshold is None:
            entropy_values: list[float] = []
            for index, record in enumerate(records):
                scores = record.get("scores", {})
                if source_score_field not in scores:
                    raise ValueError(
                        f"record {index}: scores.{source_score_field} is missing"
                    )
                entropy_values.append(scores[source_score_field])
            label_threshold = best_split_threshold(entropy_values)
            threshold_was_derived = True
        elif isinstance(label_threshold, bool) or not math.isfinite(
            float(label_threshold)
        ):
            raise ValueError("label_threshold must be a finite number")
        training_target = _common_entropy_target_provenance(
            records, role="training"
        )
        supplied_target = (
            metadata.get("entropy_target_provenance")
            if isinstance(metadata, Mapping)
            else None
        )
        target_provenance = validate_entropy_target_provenance(
            supplied_target,
            role="training metadata.entropy_target_provenance",
        )
        for field in _ENTROPY_TARGET_SHARED_FIELDS:
            if target_provenance[field] != training_target[field]:
                raise ValueError(
                    "training entropy-target records do not match serialized "
                    f"provenance for {field}"
                )
        if (
            target_provenance["train_source_artifact_sha256"]
            != training_target["source_artifact_sha256"]
        ):
            raise ValueError(
                "training entropy-target source artifact does not match "
                "serialized provenance"
            )
    elif label_source != "correctness":
        raise ValueError(
            f"unknown label_source: {label_source!r}; "
            "expected 'correctness' or 'semantic_entropy_threshold'"
        )

    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:  # pragma: no cover - covered by deps install
        raise ImportError(
            "SEP training requires numpy and scikit-learn. Install with: "
            "pip install -e '.[sep]'"
        ) from exc

    features: list[list[float]] = []
    labels: list[int] = []
    hidden_dim: int | None = None
    sep_feature_identity: dict[str, Any] | None = None
    accuracy_uses_singular_feature: bool | None = None
    for index, record in enumerate(records):
        if sep_feature_unavailable(record):
            # No probe feature at all: greedy decoding never terminated and no
            # feature was recovered, so this prompt cannot be a training
            # example. It still contributes its stochastic samples to the
            # entropy analysis. Records carrying a declared feature_backfill do
            # have a usable feature and are kept.
            continue
        if label_source == "semantic_entropy_threshold":
            hidden = record.get("sep_hidden_state")
            if hidden is None:
                raise ValueError(
                    f"record {index}: genuine SEP requires 'sep_hidden_state' "
                    "from a separate greedy response"
                )
            feature = np.asarray(hidden, dtype=float)
            if feature.ndim != 1 or feature.shape[0] == 0:
                raise ValueError(
                    f"record {index}: sep_hidden_state must be a non-empty 1D "
                    f"vector; got shape {feature.shape}"
                )
            feature_metadata = record.get("sep_feature_metadata")
            if not isinstance(feature_metadata, Mapping):
                raise ValueError(
                    f"record {index}: sep_feature_metadata must be a mapping"
                )
            record_identity = _sep_feature_identity(
                feature_metadata, role=f"training record {index}"
            )
            if record_identity["hidden_dimension"] != int(feature.shape[0]):
                raise ValueError(
                    f"record {index}: feature metadata hidden_dimension does not "
                    "match sep_hidden_state"
                )
            if sep_feature_identity is None:
                sep_feature_identity = record_identity
            elif record_identity != sep_feature_identity:
                raise ValueError(
                    f"record {index}: SEP feature identity differs from the first "
                    "training record"
                )
        else:
            singular = record.get("sep_hidden_state")
            if singular is not None:
                if accuracy_uses_singular_feature is False:
                    raise ValueError(
                        "accuracy-probe training cannot mix singular greedy "
                        "features with legacy per-sample hidden states"
                    )
                accuracy_uses_singular_feature = True
                feature = np.asarray(singular, dtype=float)
                if feature.ndim != 1 or feature.shape[0] == 0:
                    raise ValueError(
                        f"record {index}: sep_hidden_state must be a non-empty "
                        f"1D vector; got shape {feature.shape}"
                    )
                feature_metadata = record.get("sep_feature_metadata")
                if not isinstance(feature_metadata, Mapping):
                    raise ValueError(
                        f"record {index}: sep_feature_metadata must be a mapping"
                    )
                record_identity = _sep_feature_identity(
                    feature_metadata, role=f"training record {index}"
                )
                if record_identity["hidden_dimension"] != int(feature.shape[0]):
                    raise ValueError(
                        f"record {index}: feature metadata hidden_dimension does "
                        "not match sep_hidden_state"
                    )
                if sep_feature_identity is None:
                    sep_feature_identity = record_identity
                elif record_identity != sep_feature_identity:
                    raise ValueError(
                        f"record {index}: SEP feature identity differs from the "
                        "first training record"
                    )
            else:
                if accuracy_uses_singular_feature is True:
                    raise ValueError(
                        "accuracy-probe training cannot mix singular greedy "
                        "features with legacy per-sample hidden states"
                    )
                accuracy_uses_singular_feature = False
                hidden = record.get("hidden_states")
                if hidden is None:
                    raise ValueError(
                        f"record {index}: missing hidden-state feature"
                    )
                arr = np.asarray(hidden, dtype=float)
                if arr.ndim != 2:
                    raise ValueError(
                        f"record {index}: hidden_states must be 2D "
                        f"(num_samples, hidden_dim); got shape {arr.shape}"
                    )
                feature = _pool_hidden_states(
                    arr, pooling=pooling, mean_last_k=mean_last_k
                )
                if feature is None:
                    if aggregate == "first":
                        feature = arr[0]
                    elif aggregate == "mean":
                        feature = arr.mean(axis=0)
                    else:
                        raise ValueError(
                            f"unknown aggregate: {aggregate!r}; expected "
                            "'first' or 'mean'"
                        )

        if hidden_dim is None:
            hidden_dim = int(feature.shape[0])
        elif int(feature.shape[0]) != hidden_dim:
            raise ValueError(
                f"record {index}: hidden_dim {feature.shape[0]} does not match "
                f"the first record's hidden_dim {hidden_dim}"
            )
        if not np.isfinite(feature).all():
            raise ValueError(f"record {index}: hidden-state feature must be finite")
        features.append(feature.tolist())

        if label_source == "correctness":
            correctness = record.get("correctness_label")
            if not isinstance(correctness, bool):
                raise ValueError(
                    f"record {index}: correctness_label must be a bool for "
                    "label_source='correctness'"
                )
            # Positive class = uncertain = incorrect, matching the metrics
            # module's AUROC convention.
            labels.append(0 if correctness else 1)
        else:
            scores = record.get("scores", {})
            if source_score_field not in scores:
                raise ValueError(
                    f"record {index}: scores.{source_score_field} is missing"
                )
            value = scores[source_score_field]
            labels.append(1 if value >= label_threshold else 0)

    x_train = np.asarray(features, dtype=float)
    y_train = np.asarray(labels, dtype=int)

    if hidden_dim is None:
        raise ValueError("could not infer hidden_dim from records")
    if x_train.shape[0] < 2:
        raise ValueError("need at least two records to fit a probe")
    if len(set(y_train.tolist())) < 2:
        raise ValueError(
            "training labels are degenerate (only one class observed); "
            "either choose a different label_source/threshold or supply "
            "records with both correct and incorrect labels"
        )

    # Farquhar et al.'s get_p_ik fits a bare LogisticRegression(), i.e. the
    # same l2 penalty, C=1.0 and lbfgs solver used here, but leaves max_iter at
    # the scikit-learn default of 100 and records whether the fit converged
    # rather than raising the cap. The higher cap here only removes the
    # possibility of reporting an unconverged fit; convergence is recorded the
    # same way so the difference is visible instead of assumed.
    model = LogisticRegression(
        penalty="l2",
        C=C,
        solver="lbfgs",
        max_iter=2000,
        random_state=seed,
    )
    model.fit(x_train, y_train)

    # Test doubles for the estimator need not expose solver diagnostics.
    raw_n_iter = getattr(model, "n_iter_", None)
    max_iter = int(getattr(model, "max_iter", 0) or 0)
    if raw_n_iter is None or not max_iter:
        convergence: dict[str, int | bool | None] = {
            "n_iter": None,
            "converged": None,
            "converged_within_paper_max_iter": None,
            "max_iter": max_iter or None,
        }
    else:
        n_iter = int(raw_n_iter[0])
        convergence = {
            "n_iter": n_iter,
            "converged": bool(n_iter < max_iter),
            # Their get_p_ik leaves max_iter at the scikit-learn default of 100
            # and reports convergence against it. Recorded so a fit that would
            # not have converged under their configuration is visible.
            "converged_within_paper_max_iter": bool(n_iter < 100),
            "max_iter": max_iter,
        }

    coef = model.coef_[0].astype(float).tolist()
    intercept = float(model.intercept_[0])

    artifact_metadata = dict(metadata or {})
    if target_provenance is not None:
        artifact_metadata["entropy_target_provenance"] = target_provenance
    artifact_metadata.update(
        {
            "solver_convergence": convergence,
            "artifact_schema": CANONICAL_SEP_ARTIFACT_SCHEMA,
            "target_definition": (
                "high_semantic_entropy"
                if label_source == "semantic_entropy_threshold"
                else "model_incorrectness"
            ),
            "feature_definition": (
                CANONICAL_SEP_FEATURE_DEFINITION
                if (
                    label_source == "semantic_entropy_threshold"
                    or accuracy_uses_singular_feature is True
                )
                else "accuracy_probe_hidden_state"
            ),
            "threshold_selection": (
                "eq5_best_split_training_records_only"
                if threshold_was_derived
                else (
                    "caller_supplied"
                    if label_source == "semantic_entropy_threshold"
                    else "not_applicable"
                )
            ),
            "probe_fit_scope": artifact_metadata.get(
                "probe_fit_scope", "unspecified"
            ),
            "threshold_fit_scope": (
                artifact_metadata.get("threshold_fit_scope", "unspecified")
                if label_source == "semantic_entropy_threshold"
                else "not_applicable"
            ),
            "feature_aggregate": aggregate,
            "inverse_l2_regularization": float(C),
            "training_seed": int(seed),
        }
    )
    if (
        label_source == "semantic_entropy_threshold"
        or accuracy_uses_singular_feature is True
    ):
        if sep_feature_identity is None:  # defensive: records is non-empty
            raise ValueError("could not infer SEP feature identity")
        artifact_metadata["feature_identity"] = sep_feature_identity
        serialized_cell_identity = artifact_metadata.get("cell_identity")
        if serialized_cell_identity is not None:
            expected_cell_identity = validate_sep_cell_identity(
                serialized_cell_identity,
                role="training metadata.cell_identity",
            )
            actual_cell_identity = canonical_sep_cell_identity(
                records, role="training"
            )
            if actual_cell_identity != expected_cell_identity:
                raise ValueError(
                    "training records do not match serialized SEP cell_identity"
                )
            artifact_metadata["cell_identity"] = expected_cell_identity

    return SEPProbe(
        coef=coef,
        intercept=intercept,
        hidden_dim=hidden_dim,
        label_source=label_source,
        probe_kind=(
            "semantic_entropy_probe"
            if label_source == "semantic_entropy_threshold"
            else "accuracy_probe"
        ),
        label_threshold=label_threshold,
        source_score_field=source_score_field,
        train_size=int(x_train.shape[0]),
        positive_rate=float(y_train.mean()),
        metadata=artifact_metadata,
        pooling=pooling,
        mean_last_k=mean_last_k,
        artifact_schema=CANONICAL_SEP_ARTIFACT_SCHEMA,
    )


def score_probe(
    hidden_state: Sequence[float],
    probe: SEPProbe,
) -> float:
    """Return ``p(uncertain | hidden_state)`` from a trained probe.

    Args:
        hidden_state: A single hidden state vector with length
            ``probe.hidden_dim``.
        probe: A trained :class:`SEPProbe`.

    Returns:
        The probe's predicted probability of the uncertain class, in
        ``[0, 1]``.

    Raises:
        ValueError: If the input dimensionality does not match the probe.
        ImportError: If ``numpy`` is not installed.
    """
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - covered by deps install
        raise ImportError(
            "SEP scoring requires numpy. Install with: pip install -e '.[sep]'"
        ) from exc

    arr = np.asarray(hidden_state, dtype=float)
    if arr.ndim != 1 or arr.shape[0] != probe.hidden_dim:
        raise ValueError(
            f"hidden_state must be 1D of length {probe.hidden_dim}; "
            f"got shape {arr.shape}"
        )

    coef = np.asarray(probe.coef, dtype=float)
    logit = float(arr @ coef) + probe.intercept
    # Numerically stable sigmoid.
    if logit >= 0:
        z = np.exp(-logit)
        return float(1.0 / (1.0 + z))
    z = np.exp(logit)
    return float(z / (1.0 + z))


def score_probe_for_record(
    record: Mapping[str, Any],
    probe: SEPProbe,
    *,
    aggregate: Literal["first", "mean"] = "first",
) -> float | None:
    """Score a full scored-JSONL record using a trained probe.

    A genuine SEP consumes only ``sep_hidden_state`` from the separate greedy
    response. An accuracy probe may consume that same response feature or a
    legacy per-sample ``hidden_states`` matrix.

    Args:
        record: A scored record carrying a singular SEP feature or legacy
            per-sample hidden-state matrix.
        probe: A trained probe.
        aggregate: How to reduce per-sample hidden states into a single
            feature vector.  Should match the value passed to
            :func:`train_probe`.

    Returns:
        ``p(uncertain)`` for the record, as in :func:`score_probe`.
    """
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - covered by deps install
        raise ImportError(
            "SEP scoring requires numpy. Install with: pip install -e '.[sep]'"
        ) from exc

    if sep_feature_unavailable(record):
        # No feature to score. The record drops out of the probe evaluation
        # while remaining in the entropy analysis. A record whose feature was
        # recovered declares feature_backfill and is scored normally.
        return None

    probe_kind = probe.effective_probe_kind
    if probe_kind == "semantic_entropy_probe":
        validate_canonical_sep_probe(probe)
        hidden_state = record.get("sep_hidden_state")
        if hidden_state is None:
            raise ValueError(
                "genuine SEP scoring requires 'sep_hidden_state' from a "
                "separate greedy response"
            )
        feature_metadata = record.get("sep_feature_metadata")
        if not isinstance(feature_metadata, Mapping):
            raise ValueError(
                "SEP scoring requires sep_feature_metadata"
            )
        actual_identity = _sep_feature_identity(
            feature_metadata, role="scoring record"
        )
        if actual_identity != probe.metadata["feature_identity"]:
            raise ValueError(
                "scoring record SEP feature identity does not match the trained probe"
            )
        feature = np.asarray(hidden_state, dtype=float)
        if feature.ndim != 1:
            raise ValueError(
                f"sep_hidden_state must be 1D; got shape {feature.shape}"
            )
        return score_probe(feature.tolist(), probe)

    hidden_state = record.get("sep_hidden_state")
    if hidden_state is not None:
        if "feature_identity" in probe.metadata:
            validate_canonical_accuracy_probe(probe)
            feature_metadata = record.get("sep_feature_metadata")
            if not isinstance(feature_metadata, Mapping):
                raise ValueError(
                    "canonical accuracy-probe scoring requires "
                    "sep_feature_metadata"
                )
            actual_identity = _sep_feature_identity(
                feature_metadata, role="accuracy-probe scoring record"
            )
            if actual_identity != probe.metadata["feature_identity"]:
                raise ValueError(
                    "accuracy-probe scoring record feature identity does not "
                    "match the trained probe"
                )
        feature = np.asarray(hidden_state, dtype=float)
        if feature.ndim != 1:
            raise ValueError(
                f"sep_hidden_state must be 1D; got shape {feature.shape}"
            )
        return score_probe(feature.tolist(), probe)

    hidden = record.get("hidden_states")
    if hidden is None:
        raise ValueError("record is missing a hidden-state feature")

    arr = np.asarray(hidden, dtype=float)
    if arr.ndim != 2:
        raise ValueError(
            f"hidden_states must be 2D; got shape {arr.shape}"
        )

    # Use the probe's stored pooling strategy (new) falling back to the
    # legacy aggregate keyword for backward compatibility with probes that
    # pre-date the pooling field.
    probe_pooling = getattr(probe, "pooling", "last")
    probe_mean_last_k = getattr(probe, "mean_last_k", 8)

    pooled = _pool_hidden_states(arr, pooling=probe_pooling, mean_last_k=probe_mean_last_k)
    if pooled is None:
        # pooling == "last": fall back to aggregate keyword
        if aggregate == "first":
            feature = arr[0]
        elif aggregate == "mean":
            feature = arr.mean(axis=0)
        else:
            raise ValueError(
                f"unknown aggregate: {aggregate!r}; expected 'first' or 'mean'"
            )
    else:
        feature = pooled

    return score_probe(feature.tolist(), probe)
