"""Entropy estimators for the discrete semantic-entropy baseline.

This module implements the two uncertainty signals required by the Phase 1
experimental protocol (see ``docs/experiment_protocol.md``): a *surface-form*
entropy over exact normalized answer strings, and a *discrete semantic
entropy* over the cluster assignments produced by an entailment-based
clustering step. Both quantities are computed in nats from a single sample of
``M`` generations per prompt and are therefore directly comparable on a
common scale.

The estimator used throughout is the maximum-likelihood (plug-in) estimator

    H_hat(X) = -sum_i (n_i / N) log(n_i / N),

where ``N`` is the sample size and ``n_i`` is the number of observations
falling in category ``i``. This is the form adopted by Kuhn, Gal and Farquhar
(2023) and by Farquhar et al. (2024) for their discrete semantic-entropy
baseline. It is known to be downward-biased in finite samples; bias-corrected
alternatives (Miller--Madow, NSB) are deliberately deferred to the Phase 2
extension work and are not invoked here.

Probability-weighted semantic entropy, which substitutes per-sequence
generation probabilities for empirical frequencies, is the optional Phase 2
extension noted in ``AGENTS.md`` and is intentionally not implemented in this
module.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Hashable, Sequence


def surface_entropy(normalized_answers: Sequence[str]) -> float:
    """Plug-in Shannon entropy over normalized answer strings (in nats).

    Implements the surface-form uncertainty baseline of the Phase 1 protocol:
    each unique normalized answer string is treated as its own category, and
    the entropy of the empirical categorical distribution is returned. The
    function makes no semantic equivalence judgement -- strings that are
    paraphrases but not character-identical after normalization are counted
    as distinct categories. This is by design, since the contrast against
    :func:`discrete_semantic_entropy` is the central comparison of the
    thesis.
    """

    return entropy_from_labels(normalized_answers)


def discrete_semantic_entropy(cluster_ids: Sequence[int]) -> float:
    """Plug-in Shannon entropy over semantic-cluster identifiers (in nats).

    Implements the Monte-Carlo / equal-weight form of discrete semantic
    entropy described by Kuhn et al. (2023) and Farquhar et al. (2024):
    each sampled generation is assigned to a meaning-cluster (typically by
    bidirectional entailment), and entropy is taken over the resulting
    empirical cluster distribution. Per-sequence generation probabilities
    are not used; doing so would yield the full probability-weighted
    semantic entropy reserved for Phase 2.
    """

    return entropy_from_labels(cluster_ids)


def score_record(
    normalized_answers: Sequence[str], cluster_ids: Sequence[int]
) -> dict[str, float]:
    """Return both Phase 1 uncertainty scores for a single prompt record.

    The two scores are computed independently on the same sample of
    generations: surface-form entropy from the normalized answer strings and
    discrete semantic entropy from the cluster assignments. They populate
    the ``scores`` sub-object of the JSONL schema in
    ``docs/experiment_protocol.md``.
    """

    return {
        "surface_entropy": surface_entropy(normalized_answers),
        "discrete_semantic_entropy": discrete_semantic_entropy(cluster_ids),
    }


def entropy_from_labels(labels: Sequence[Hashable]) -> float:
    """Plug-in Shannon entropy in nats of a categorical sample.

    Given a finite, non-empty sequence of hashable labels, this function
    returns ``H_hat = -sum (n_i / N) log(n_i / N)``, the maximum-likelihood
    estimator of the entropy of the categorical distribution that generated
    the sample. The natural logarithm is used so that surface-form and
    discrete-semantic entropies share the same unit and are directly
    comparable. Empty input raises :class:`ValueError`, which propagates up
    through :func:`score_record` and the CLI so that malformed records fail
    loudly rather than silently producing a default value.
    """

    if not labels:
        raise ValueError("labels must not be empty")

    total = len(labels)
    counts = Counter(labels)
    return -sum((count / total) * math.log(count / total) for count in counts.values())
