"""Entropy estimators for the discrete semantic-entropy baseline.

This module implements the uncertainty signals used in the experiments: a naive
sampled-answer entropy over exact raw generations, a *surface-form* entropy
over exact normalized answer strings, and a *discrete semantic entropy* over
the cluster assignments produced by an entailment-based clustering step. All
quantities are computed in nats from a single sample of ``M`` generations per
prompt and are therefore directly comparable on a common scale.

The estimator used throughout is the maximum-likelihood (plug-in) estimator

    H_hat(X) = -sum_i (n_i / N) log(n_i / N),

where ``N`` is the sample size and ``n_i`` is the number of observations
falling in category ``i``. This is the form adopted by Kuhn, Gal and Farquhar
(2023) and by Farquhar et al. (2024) for their discrete semantic-entropy
baseline. It is known to be downward-biased in finite samples; bias-corrected
alternatives (Miller--Madow, NSB) are deliberately deferred to the Phase 2
extension work and are not invoked here.

Taxonomy of the estimators in this module (two axes: *surface vs semantic*
and *count-based vs probability-weighted*):

                    | count-based (frequencies) | probability-weighted (logprobs)
    surface only    | naive_sample_entropy      | naive_entropy
                    | surface_entropy           |   (paper's naive baseline)
    semantic        | discrete_semantic_entropy | semantic_entropy_full
    (clustered)     |   (paper's discrete approx.) | (paper's PRIMARY estimator)

The count-based estimators (``naive_sample_entropy``, ``surface_entropy``,
``discrete_semantic_entropy``) need only a sample of generations. The
probability-weighted estimators (``naive_entropy``, ``semantic_entropy_full``)
additionally need per-sequence length-normalized log-probabilities, captured
at generation time (see ``schema.py`` ``sequence_logprobs``). ``naive_entropy``
and ``semantic_entropy_full`` reproduce Farquhar et al. (2024); the count-based
``naive_sample_entropy``/``surface_entropy`` are this project's black-box
surface baselines and are *not* the paper's naive entropy.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Hashable, Sequence


def naive_sample_entropy(sampled_answers: Sequence[str]) -> float:
    """Plug-in Shannon entropy over exact raw sampled answers (in nats).

    This is the lexical diversity baseline before normalization or semantic
    clustering: every distinct sampled answer string is treated as a separate
    category, including differences in casing, spacing, or wording.
    """

    return entropy_from_labels(sampled_answers)


def surface_entropy(normalized_answers: Sequence[str]) -> float:
    """Plug-in Shannon entropy over normalized answer strings (in nats).

    Implements the surface-form uncertainty baseline of the Phase 1 protocol:
    each unique normalized answer string is treated as its own category, and
    the entropy of the empirical categorical distribution is returned. The
    function makes no semantic equivalence judgment -- strings that are
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


def naive_entropy(sequence_logprobs: Sequence[float]) -> float:
    """Probability-based naive (predictive) entropy in nats.

    This is the paper's *naive entropy* baseline (Farquhar et al. 2024): a
    Monte-Carlo estimate of the predictive entropy over the model's own
    generation distribution, with **no** semantic clustering::

        H(Y | x) ~= -(1/M) * sum_i log p(s_i | x)

    where ``log p(s_i | x)`` is the *length-normalized* joint log-probability
    of sampled sequence ``s_i`` (arithmetic mean token log-prob). Unlike
    :func:`naive_sample_entropy`, which counts exact-string frequencies and is
    bounded by ``log M``, this estimator uses the generation probabilities and
    is unbounded above. The two are different quantities and not comparable.

    Args:
        sequence_logprobs: One length-normalized log-probability per sampled
            sequence (``M`` values). Must be non-empty.
    """

    if not sequence_logprobs:
        raise ValueError("sequence_logprobs must not be empty")
    return -math.fsum(sequence_logprobs) / len(sequence_logprobs)


def semantic_entropy_full(
    cluster_ids: Sequence[int], sequence_logprobs: Sequence[float]
) -> float:
    """Full probability-weighted semantic entropy in nats.

    This is the paper's *primary* estimator (Kuhn et al. 2023; Farquhar et al.
    2024). Sequences are grouped into meaning-clusters, each cluster's
    probability is the sum of its members' (length-normalized) sequence
    probabilities, the cluster probabilities are renormalized to sum to one,
    and Shannon entropy is taken over them::

        log P(c | x) = logsumexp_{s in c} log p(s | x)
        P(c | x)     = P(c | x) / sum_c' P(c' | x)
        SE           = - sum_c P(c | x) log P(c | x)

    In the limit where all sequence log-probabilities are equal, this collapses
    to :func:`discrete_semantic_entropy` (the count-based approximation), which
    is asserted as a parity test.

    Args:
        cluster_ids: Meaning-cluster id per sampled sequence.
        sequence_logprobs: Length-normalized log-probability per sequence,
            aligned with ``cluster_ids``. Both must be non-empty and equal
            length.
    """

    if not cluster_ids or not sequence_logprobs:
        raise ValueError("cluster_ids and sequence_logprobs must not be empty")
    if len(cluster_ids) != len(sequence_logprobs):
        raise ValueError("cluster_ids and sequence_logprobs must have equal length")

    logp_by_cluster: dict[int, list[float]] = {}
    for cid, logp in zip(cluster_ids, sequence_logprobs):
        logp_by_cluster.setdefault(cid, []).append(logp)

    cluster_logp = {cid: _logsumexp(values) for cid, values in logp_by_cluster.items()}
    log_total = _logsumexp(list(cluster_logp.values()))
    probs = [math.exp(lp - log_total) for lp in cluster_logp.values()]
    return -math.fsum(p * math.log(p) for p in probs if p > 0.0)


def _logsumexp(values: Sequence[float]) -> float:
    """Numerically stable log(sum(exp(values)))."""
    if not values:
        raise ValueError("values must not be empty")
    peak = max(values)
    if peak == -math.inf:
        return -math.inf
    return peak + math.log(math.fsum(math.exp(v - peak) for v in values))


def score_record(
    sampled_answers: Sequence[str],
    normalized_answers: Sequence[str],
    cluster_ids: Sequence[int],
    sequence_logprobs: Sequence[float] | None = None,
) -> dict[str, float]:
    """Return uncertainty scores for a single prompt record.

    Always computes the three count-based estimators (naive sample entropy
    over raw strings, surface entropy over normalized strings, discrete
    semantic entropy over clusters). When ``sequence_logprobs`` are supplied
    (captured at generation time), it additionally computes the two
    probability-weighted estimators (``naive_entropy`` and
    ``semantic_entropy_full``). They populate the ``scores`` sub-object of the
    JSONL schema validated by ``semantic_entropy.schema``.
    """

    scores = {
        "naive_sample_entropy": naive_sample_entropy(sampled_answers),
        "surface_entropy": surface_entropy(normalized_answers),
        "discrete_semantic_entropy": discrete_semantic_entropy(cluster_ids),
    }
    if sequence_logprobs is not None and len(sequence_logprobs) > 0:
        scores["naive_entropy"] = naive_entropy(sequence_logprobs)
        scores["semantic_entropy_full"] = semantic_entropy_full(
            cluster_ids, sequence_logprobs
        )
    return scores


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
