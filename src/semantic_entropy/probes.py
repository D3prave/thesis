"""Semantic Entropy Probes (SEP): linear probes over hidden states.

SEP trains a small linear classifier on a generation model's *internal*
hidden states to predict whether the generation is uncertain — proxied
either by the correctness label, or by a binarization of an already-computed
semantic entropy score.  Reference: Slobodkin et al. (2023), "The Curious
Case of Hallucinatory (Un)answerability"; probing baseline in Farquhar
et al. (2024).

The motivation is test-time cost: discrete semantic entropy requires
sampling ``M = 10`` generations per prompt plus pairwise entailment, while
SEP requires only a single forward pass per prompt.  When the probe
generalises, it gives a cheap stand-in for the full entropy pipeline.

This module provides two top-level functions and a serialisation-friendly
dataclass:

* :func:`train_probe` — fit an L2-regularised logistic regression from a
  set of scored records that include per-sample hidden states.
* :func:`score_probe` — return ``p(uncertain)`` in ``[0, 1]`` for a fresh
  hidden state vector.
* :class:`SEPProbe` — dataclass holding the trained coefficients and
  enough metadata to reconstruct the probe from JSON.

The training routine accepts records produced by the run-pipeline when
invoked with a ``ModelFnWithStates`` adapter (see
:mod:`semantic_entropy.models`).  Each record must carry a
``hidden_states`` array of shape ``(num_samples, hidden_dim)``; the probe
labels can either be ``correctness_label`` (already a per-record bool) or
the *binarized* version of an existing entropy score using a
caller-supplied threshold.

``numpy`` and ``scikit-learn`` are deferred imports so the rest of the
package keeps its stdlib-only runtime contract.  Installing the optional
extras (``pip install -e '.[sep]'``) brings both in.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SEPProbe:
    """Serialisable container for a trained SEP linear probe.

    The probe is a single-layer logistic regression
    ``p(uncertain | h) = sigmoid(coef . h + intercept)`` over a hidden
    state vector ``h`` of dimensionality ``hidden_dim``.

    Attributes:
        coef: One coefficient per input dimension.  Length must equal
            ``hidden_dim``.
        intercept: Scalar bias term.
        hidden_dim: Expected dimensionality of the input hidden state.
        label_source: Either ``"correctness"`` (probe trained from
            ``correctness_label``, *uncertain = incorrect*) or
            ``"semantic_entropy_threshold"`` (probe trained from a
            binarized entropy score).
        label_threshold: For ``label_source == "semantic_entropy_threshold"``,
            the threshold that was used to binarize the source entropy.
            ``None`` otherwise.
        source_score_field: For ``label_source == "semantic_entropy_threshold"``,
            the name of the entropy score that was binarized.  ``None``
            otherwise.
        train_size: Number of (sample, label) pairs used to fit the probe.
        positive_rate: Empirical fraction of *uncertain* labels in the
            training set.  Useful for sanity-checking class balance.
    """

    coef: list[float]
    intercept: float
    hidden_dim: int
    label_source: Literal["correctness", "semantic_entropy_threshold"]
    label_threshold: float | None = None
    source_score_field: str | None = None
    train_size: int = 0
    positive_rate: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the probe."""
        return {
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
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SEPProbe":
        """Reconstruct a probe from :meth:`to_dict` output."""
        return cls(
            coef=[float(c) for c in payload["coef"]],
            intercept=float(payload["intercept"]),
            hidden_dim=int(payload["hidden_dim"]),
            label_source=payload["label_source"],
            label_threshold=(
                None
                if payload.get("label_threshold") is None
                else float(payload["label_threshold"])
            ),
            source_score_field=payload.get("source_score_field"),
            train_size=int(payload.get("train_size", 0)),
            positive_rate=float(payload.get("positive_rate", 0.0)),
            metadata=dict(payload.get("metadata", {})),
        )

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


def train_probe(
    records: Sequence[Mapping[str, Any]],
    *,
    label_source: Literal["correctness", "semantic_entropy_threshold"] = "correctness",
    source_score_field: str | None = None,
    label_threshold: float | None = None,
    aggregate: Literal["first", "mean"] = "first",
    C: float = 1.0,
    seed: int = 0,
) -> SEPProbe:
    """Fit a logistic-regression probe on per-record hidden states.

    Args:
        records: Iterable of scored records.  Each record must contain a
            ``hidden_states`` field shaped ``(num_samples, hidden_dim)``
            and the relevant label field (``correctness_label`` for
            ``label_source="correctness"`` or
            ``scores[source_score_field]`` for the threshold source).
        label_source: Defines what the probe predicts.

            * ``"correctness"`` — the positive class is *incorrect*
              (``not correctness_label``), matching the AUROC convention
              used throughout the metrics module.  This is the canonical
              SEP target.
            * ``"semantic_entropy_threshold"`` — the positive class is
              ``scores[source_score_field] >= label_threshold``.  Useful
              when you want the probe to mimic an entropy estimator
              rather than the correctness label.

        source_score_field: Name of the entropy score under
            ``records[i]["scores"]`` to binarize.  Required when
            ``label_source == "semantic_entropy_threshold"``.
        label_threshold: Threshold for the entropy binarization.
            Required when ``label_source == "semantic_entropy_threshold"``.
        aggregate: How to convert the per-sample hidden states of a single
            record into one feature vector for training.

            * ``"first"`` (default) — use the hidden state of the first
              sampled generation.  This matches the SEP inference path,
              where only a single forward pass is performed per prompt.
            * ``"mean"`` — average the per-sample hidden states.  This is
              useful when the probe is trained from records that already
              have ``M`` samples and we want to give the trainer the
              denoised mean.

        C: Inverse-regularisation strength forwarded to
            ``sklearn.linear_model.LogisticRegression``.  Larger means
            weaker L2 regularisation.
        seed: Random seed forwarded to the scikit-learn solver.

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

    if label_source == "semantic_entropy_threshold":
        if source_score_field is None or label_threshold is None:
            raise ValueError(
                "label_source='semantic_entropy_threshold' requires both "
                "source_score_field and label_threshold to be set"
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
    for index, record in enumerate(records):
        hidden = record.get("hidden_states")
        if hidden is None:
            raise ValueError(
                f"record {index}: missing 'hidden_states' field; was the "
                "model_fn_with_states adapter used during sampling?"
            )
        arr = np.asarray(hidden, dtype=float)
        if arr.ndim != 2:
            raise ValueError(
                f"record {index}: hidden_states must be 2D "
                f"(num_samples, hidden_dim); got shape {arr.shape}"
            )
        if hidden_dim is None:
            hidden_dim = int(arr.shape[1])
        elif int(arr.shape[1]) != hidden_dim:
            raise ValueError(
                f"record {index}: hidden_dim {arr.shape[1]} does not match "
                f"the first record's hidden_dim {hidden_dim}"
            )

        if aggregate == "first":
            feature = arr[0]
        elif aggregate == "mean":
            feature = arr.mean(axis=0)
        else:
            raise ValueError(
                f"unknown aggregate: {aggregate!r}; expected 'first' or 'mean'"
            )
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

    model = LogisticRegression(
        penalty="l2",
        C=C,
        solver="lbfgs",
        max_iter=2000,
        random_state=seed,
    )
    model.fit(x_train, y_train)

    coef = model.coef_[0].astype(float).tolist()
    intercept = float(model.intercept_[0])

    return SEPProbe(
        coef=coef,
        intercept=intercept,
        hidden_dim=hidden_dim,
        label_source=label_source,
        label_threshold=label_threshold,
        source_score_field=source_score_field,
        train_size=int(x_train.shape[0]),
        positive_rate=float(y_train.mean()),
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
) -> float:
    """Score a full scored-JSONL record using a trained probe.

    Convenience wrapper that mirrors the aggregation choice made during
    training: it picks either the first sample's hidden state or the
    mean across samples, then forwards to :func:`score_probe`.

    Args:
        record: A scored record carrying ``hidden_states`` shaped
            ``(num_samples, hidden_dim)``.
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

    hidden = record.get("hidden_states")
    if hidden is None:
        raise ValueError("record is missing 'hidden_states' field")

    arr = np.asarray(hidden, dtype=float)
    if arr.ndim != 2:
        raise ValueError(
            f"hidden_states must be 2D; got shape {arr.shape}"
        )

    if aggregate == "first":
        feature = arr[0]
    elif aggregate == "mean":
        feature = arr.mean(axis=0)
    else:
        raise ValueError(
            f"unknown aggregate: {aggregate!r}; expected 'first' or 'mean'"
        )

    return score_probe(feature.tolist(), probe)
