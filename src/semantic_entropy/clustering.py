"""Semantic clustering for the Phase 1 pipeline.

This module provides two clustering functions that sit between answer
normalization and entropy scoring, each taking a list of normalized answer
strings and returning a ``(cluster_ids, representatives)`` pair.

**:func:`exact_match_cluster`** (baseline stub)
    Groups answers by exact string equality. Requires no external model and
    is used as the default when no NLI backend is supplied. It intentionally
    under-clusters paraphrases and is kept for smoke tests, rapid iteration,
    and as the default fallback in :func:`~semantic_entropy.harness.sample_record`.

**:func:`nli_cluster`** (Stage 3 semantic clustering)
    Implements the bidirectional-entailment clustering algorithm described
    by Kuhn et al. (2023) and Farquhar et al. (2024): two answers are placed
    in the same cluster if and only if they *mutually* entail each other
    under the supplied NLI function. Connected components are found via
    union-find, so transitivity is handled automatically — if A entails B
    and B entails C, all three land in the same cluster even if A and C are
    never directly compared.

    The :data:`NliFn` type alias specifies the expected interface: a callable
    that takes a ``(premise, hypothesis)`` string pair and returns one of the
    string constants :data:`NLI_ENTAILMENT`, :data:`NLI_NEUTRAL`, or
    :data:`NLI_CONTRADICTION`. The :func:`exact_match_entailment_fn` stub
    satisfies this interface by treating equal strings as mutually entailing,
    reproducing :func:`exact_match_cluster` exactly.

    To use a real NLI model (e.g. ``cross-encoder/nli-deberta-v3-base``),
    wrap it in a function matching :data:`NliFn` and pass it to
    :func:`~semantic_entropy.harness.sample_record` or
    :func:`~semantic_entropy.harness.run_pipeline`.

**Cluster ID invariants** (enforced by both functions and checked by
:func:`semantic_entropy.cluster_check.check_cluster_consistency`):

* IDs are non-negative integers starting from 0.
* IDs are *contiguous*: if ``k`` distinct clusters are observed, the IDs
  used are exactly ``{0, 1, …, k-1}``.
* IDs are assigned in *first-occurrence order*: the cluster containing the
  first answer in the list receives ID 0, the next new cluster encountered
  left-to-right receives ID 1, and so on.
* There is exactly one representative per cluster, taken as the first
  answer string assigned to that cluster.
"""

from __future__ import annotations

from collections.abc import Callable


# ---------------------------------------------------------------------------
# NLI label constants and type alias
# ---------------------------------------------------------------------------

NLI_ENTAILMENT: str = "entailment"
NLI_NEUTRAL: str = "neutral"
NLI_CONTRADICTION: str = "contradiction"

#: Type alias for a natural-language inference function.
#:
#: A ``NliFn`` takes a *premise* string and a *hypothesis* string and returns
#: one of :data:`NLI_ENTAILMENT`, :data:`NLI_NEUTRAL`, or
#: :data:`NLI_CONTRADICTION`. For semantic clustering only ``NLI_ENTAILMENT``
#: is acted upon; the other two labels are treated identically (no merge).
#:
#: Example wrapping a HuggingFace cross-encoder::
#:
#:     from transformers import pipeline
#:     _nli = pipeline("text-classification",
#:                     model="cross-encoder/nli-deberta-v3-base")
#:
#:     def deberta_entailment_fn(premise: str, hypothesis: str) -> str:
#:         result = _nli(f"{premise} [SEP] {hypothesis}")[0]
#:         label = result["label"].lower()  # "entailment", "neutral", "contradiction"
#:         return label
NliFn = Callable[[str, str], str]


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


# ---------------------------------------------------------------------------
# NLI entailment stub
# ---------------------------------------------------------------------------


def exact_match_entailment_fn(premise: str, hypothesis: str) -> str:
    """Return ``NLI_ENTAILMENT`` iff *premise* and *hypothesis* are identical.

    This is the trivial :data:`NliFn` implementation that reproduces
    :func:`exact_match_cluster` behavior when passed to :func:`nli_cluster`.
    It requires no ML model and is the default used by
    :func:`~semantic_entropy.harness.sample_record` until a real NLI backend
    is supplied.

    Args:
        premise: First normalized answer string.
        hypothesis: Second normalized answer string.

    Returns:
        :data:`NLI_ENTAILMENT` if the strings are equal,
        :data:`NLI_NEUTRAL` otherwise.
    """
    return NLI_ENTAILMENT if premise == hypothesis else NLI_NEUTRAL


# ---------------------------------------------------------------------------
# Bidirectional NLI clustering
# ---------------------------------------------------------------------------


def nli_cluster(
    normalized_answers: list[str],
    entailment_fn: NliFn,
) -> tuple[list[int], list[str]]:
    """Assign cluster IDs via bidirectional-entailment connected components.

    Two answers ``i`` and ``j`` are merged into the same cluster if and only
    if ``entailment_fn(answers[i], answers[j]) == NLI_ENTAILMENT`` **and**
    ``entailment_fn(answers[j], answers[i]) == NLI_ENTAILMENT``. Transitivity
    is handled automatically by union-find: if A↔B and B↔C, then A, B, and C
    all land in the same cluster even if A and C are never directly compared.

    This implements the Monte-Carlo discrete semantic-entropy clustering
    described by Kuhn et al. (2023) and Farquhar et al. (2024). When
    *entailment_fn* is :func:`exact_match_entailment_fn` the result is
    identical to :func:`exact_match_cluster`.

    Cluster ID invariants are identical to those of :func:`exact_match_cluster`:
    non-negative, contiguous, assigned in first-occurrence order, one
    representative per cluster (the first-seen answer string in that cluster).

    Args:
        normalized_answers: A non-empty list of normalized answer strings.
        entailment_fn: NLI function satisfying the :data:`NliFn` contract.
            Called at most ``n*(n-1)`` times for ``n`` answers (every ordered
            pair); a real NLI model should therefore be batched externally if
            ``n`` is large.

    Returns:
        A 2-tuple ``(semantic_clusters, cluster_representatives)`` with the
        same structure and invariants as :func:`exact_match_cluster`.

    Raises:
        ValueError: If *normalized_answers* is empty.

    Examples:
        >>> nli_cluster(["paris", "paris", "lyon"], exact_match_entailment_fn)
        ([0, 0, 1], ['paris', 'lyon'])

        >>> def liberal(p, h): return "entailment"  # everything entails everything
        >>> nli_cluster(["paris", "lyon", "berlin"], liberal)
        ([0, 0, 0], ['paris'])
    """
    if not normalized_answers:
        raise ValueError("normalized_answers must not be empty")

    n = len(normalized_answers)

    # --- Union-Find with path compression -----------------------------------
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Merge bidirectionally entailing pairs.
    for i in range(n):
        for j in range(i + 1, n):
            if (
                entailment_fn(normalized_answers[i], normalized_answers[j])
                == NLI_ENTAILMENT
                and entailment_fn(normalized_answers[j], normalized_answers[i])
                == NLI_ENTAILMENT
            ):
                union(i, j)

    # --- Assign contiguous IDs in first-occurrence order --------------------
    root_to_id: dict[int, int] = {}
    cluster_ids: list[int] = []
    for i in range(n):
        root = find(i)
        if root not in root_to_id:
            root_to_id[root] = len(root_to_id)
        cluster_ids.append(root_to_id[root])

    # Representatives: first-seen answer per cluster.
    representatives: list[str] = [""] * len(root_to_id)
    seen_cids: set[int] = set()
    for i, cid in enumerate(cluster_ids):
        if cid not in seen_cids:
            representatives[cid] = normalized_answers[i]
            seen_cids.add(cid)

    return cluster_ids, representatives
