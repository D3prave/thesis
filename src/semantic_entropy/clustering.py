"""Semantic clustering stubs for Phase 1 pipeline plumbing.

This module provides the clustering step that sits between answer
normalization and entropy scoring: it takes a list of normalized answer
strings and assigns each one a non-negative integer cluster identifier.

**Current implementation:** :func:`exact_match_cluster` groups answers
by exact string equality. Two answers belong to the same cluster if and
only if they are character-identical after normalization. This is the
fastest possible clustering and requires no ML model or network call, making
it suitable for smoke tests and local pipeline development.

**Stage 3 replacement:** exact-match clustering systematically under-
clusters paraphrases — for example, ``"william shakespeare"`` and
``"shakespeare"`` would land in different clusters even though they denote
the same entity. Stage 3 replaces this stub with bidirectional NLI-based
clustering using a DeBERTa-v3 cross-encoder (or equivalent), as specified
in ``PLANS.md`` Stage 3 and ``docs/experiment_protocol.md``. The interface
contract (inputs and outputs of :func:`exact_match_cluster`) is preserved
so that the swap is a drop-in replacement.

**Cluster ID invariants** (enforced by :func:`exact_match_cluster` and
checked by :func:`semantic_entropy.cluster_check.check_cluster_consistency`):

* IDs are non-negative integers starting from 0.
* IDs are *contiguous*: if ``k`` distinct clusters are observed, the IDs
  used are exactly ``{0, 1, …, k-1}``.
* IDs are assigned in *first-occurrence order*: the first unique normalized
  answer seen in left-to-right order receives ID 0, the next new unique
  answer receives ID 1, and so on.
* There is exactly one representative per cluster, taken as the first
  occurrence of that cluster's normalized answer string.
"""

from __future__ import annotations


def exact_match_cluster(
    normalized_answers: list[str],
) -> tuple[list[int], list[str]]:
    """Assign cluster IDs by exact normalized-string equality.

    Each unique string in *normalized_answers* is assigned a distinct
    non-negative integer cluster ID in first-occurrence order. Answers that
    are character-identical receive the same ID; answers that differ by even
    a single character receive different IDs.

    This is a *stub* implementation. It requires no external model and
    produces valid output for the downstream entropy estimators and schema
    validator, but it under-clusters paraphrases that a bidirectional NLI
    model would merge. It will be replaced in Stage 3 of the thesis pipeline.

    Args:
        normalized_answers: A non-empty list of normalized answer strings,
            one per sampled model generation. Typically produced by
            :func:`semantic_entropy.normalization.normalize_answers`.

    Returns:
        A 2-tuple ``(semantic_clusters, cluster_representatives)`` where:

        * ``semantic_clusters`` is a list of non-negative integers of the
          same length as *normalized_answers*; ``semantic_clusters[i]`` is
          the cluster ID assigned to ``normalized_answers[i]``.
        * ``cluster_representatives`` is a list of strings of length equal
          to the number of distinct clusters; ``cluster_representatives[k]``
          is the first-occurring normalized answer that was assigned cluster
          ID ``k``.

    Raises:
        ValueError: If *normalized_answers* is empty.

    Examples:
        >>> exact_match_cluster(["paris", "paris", "lyon", "paris"])
        ([0, 0, 1, 0], ['paris', 'lyon'])

        >>> exact_match_cluster(["7", "seven", "7"])
        ([0, 1, 0], ['7', 'seven'])

        >>> exact_match_cluster(["jupiter"])
        ([0], ['jupiter'])
    """
    if not normalized_answers:
        raise ValueError("normalized_answers must not be empty")

    # Map each unique answer to its cluster ID in first-occurrence order.
    seen: dict[str, int] = {}
    cluster_ids: list[int] = []
    for answer in normalized_answers:
        if answer not in seen:
            seen[answer] = len(seen)
        cluster_ids.append(seen[answer])

    # Build the representatives list indexed by cluster ID.
    representatives: list[str] = [""] * len(seen)
    for answer, cid in seen.items():
        representatives[cid] = answer

    return cluster_ids, representatives
