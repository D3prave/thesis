"""Entropy scoring utilities for the discrete semantic entropy pilot."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Hashable, Sequence


def surface_entropy(normalized_answers: Sequence[str]) -> float:
    """Compute entropy over exact normalized answer strings."""

    return entropy_from_labels(normalized_answers)


def discrete_semantic_entropy(cluster_ids: Sequence[int]) -> float:
    """Compute entropy over semantic cluster identifiers."""

    return entropy_from_labels(cluster_ids)


def score_record(
    normalized_answers: Sequence[str], cluster_ids: Sequence[int]
) -> dict[str, float]:
    """Return the two core uncertainty scores for one prompt record."""

    return {
        "surface_entropy": surface_entropy(normalized_answers),
        "discrete_semantic_entropy": discrete_semantic_entropy(cluster_ids),
    }


def entropy_from_labels(labels: Sequence[Hashable]) -> float:
    """Compute empirical entropy in nats from a sequence of labels."""

    if not labels:
        raise ValueError("labels must not be empty")

    total = len(labels)
    counts = Counter(labels)
    return -sum((count / total) * math.log(count / total) for count in counts.values())
