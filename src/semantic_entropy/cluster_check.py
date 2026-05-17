"""Internal-consistency checks for synthetic cluster assignments.

This module implements the cluster-level invariants that
:mod:`semantic_entropy.schema` deliberately leaves out. The schema validates
only the *structural* shape of a JSONL record -- types, presence, and
per-sample array cardinality. The present module validates the *semantic*
invariants of the clustering output for a synthetic record:

* Cluster identifiers are non-negative integers.
* The set of distinct identifiers is canonical, i.e. ``{0, 1, ..., K-1}``
  for some ``K``. This is the indexing convention adopted by the Nature
  semantic-entropy paper and by the released semantic-uncertainty
  codebase; non-canonical identifiers leave the alignment between cluster
  IDs and their representatives ambiguous.
* ``cluster_representatives`` contains exactly one entry per cluster,
  indexed by cluster identifier: ``cluster_representatives[k]`` is the
  representative for cluster ``k``.
* For every cluster ``k`` the representative
  ``cluster_representatives[k]`` is found among the normalized answers
  whose cluster identifier is ``k`` -- in other words, each representative
  is drawn from the members of the cluster it represents.

The checks are intentionally limited to synthetic JSONL records. They
perform no model inference, no NLI computation, no dataset loading, and
no Slurm orchestration. Violations are reported via
:class:`semantic_entropy.schema.SchemaError` with a precise field path,
mirroring the convention used by the schema validator.

A known limitation, recorded here rather than enforced: the same
normalized answer string may in principle appear in two different
clusters (e.g. if string normalization is coarser than the underlying
clustering). The present check neither requires nor forbids that
situation; it only verifies that each cluster's nominated representative
is one of its own members.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from semantic_entropy.schema import SchemaError


def check_cluster_consistency(record: Mapping[str, Any]) -> None:
    """Validate that cluster assignments and representatives agree.

    The function expects ``normalized_answers``, ``semantic_clusters``, and
    ``cluster_representatives`` to be present on the record. It performs
    the type checks needed to read those fields safely, so it may be
    called independently of :func:`semantic_entropy.schema.validate_record`,
    but in practice it is intended to be applied after structural
    validation has already passed. The function returns ``None`` on
    success and raises :class:`SchemaError` on the first violation found.
    """

    if not isinstance(record, Mapping):
        raise SchemaError("record must be a mapping")

    normalized_answers = _require_list_of_str(record, "normalized_answers")
    semantic_clusters = _require_list_of_int(record, "semantic_clusters")
    cluster_representatives = _require_list_of_str(record, "cluster_representatives")

    if len(normalized_answers) != len(semantic_clusters):
        raise SchemaError(
            "semantic_clusters length must equal normalized_answers length"
        )

    # Non-negativity: cluster identifiers index into representatives, so a
    # negative value would have no meaning.
    for index, cid in enumerate(semantic_clusters):
        if cid < 0:
            raise SchemaError(
                f"semantic_clusters[{index}] = {cid} is negative; "
                "cluster IDs must be non-negative"
            )

    # Canonicality: the distinct IDs must form a contiguous prefix
    # {0, 1, ..., K-1}. This is required so that cluster_representatives
    # can be addressed by cluster identifier without an additional lookup
    # table.
    distinct = sorted(set(semantic_clusters))
    expected = list(range(len(distinct)))
    if distinct != expected:
        raise SchemaError(
            "semantic_clusters IDs are not canonical: "
            f"expected {expected}, got distinct IDs {distinct}"
        )

    num_clusters = len(distinct)
    if len(cluster_representatives) != num_clusters:
        raise SchemaError(
            f"cluster_representatives has {len(cluster_representatives)} "
            f"entries but {num_clusters} clusters were observed"
        )

    # Bucket the normalized members by cluster identifier so that the
    # representative-in-cluster check is a single set lookup per cluster.
    members_by_cluster: dict[int, set[str]] = {k: set() for k in range(num_clusters)}
    for answer, cid in zip(normalized_answers, semantic_clusters):
        members_by_cluster[cid].add(answer)

    for k in range(num_clusters):
        representative = cluster_representatives[k]
        if representative not in members_by_cluster[k]:
            members = sorted(members_by_cluster[k])
            raise SchemaError(
                f"cluster_representatives[{k}] = {representative!r} is not "
                f"among the normalized_answers of cluster {k}: {members}"
            )


def check_jsonl_cluster_consistency(path: Path) -> int:
    """Apply :func:`check_cluster_consistency` to every record in a JSONL file.

    Returns the number of records checked. Empty input lines and JSON
    decode failures are reported with their source line number and abort
    the check, matching the strict-failure stance of the rest of the
    pipeline.
    """

    records_checked = 0
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                raise SchemaError(f"line {line_number}: empty JSONL record")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise SchemaError(
                    f"line {line_number}: invalid JSON: {error.msg}"
                ) from error
            if not isinstance(record, dict):
                raise SchemaError(
                    f"line {line_number}: record must be a JSON object"
                )
            try:
                check_cluster_consistency(record)
            except SchemaError as error:
                raise SchemaError(f"line {line_number}: {error}") from error
            records_checked += 1

    return records_checked


def _require_list_of_str(record: Mapping[str, Any], field: str) -> list[str]:
    if field not in record:
        raise SchemaError(f"{field} is required")
    value = record[field]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SchemaError(f"{field} must be a list of strings")
    return value


def _require_list_of_int(record: Mapping[str, Any], field: str) -> list[int]:
    if field not in record:
        raise SchemaError(f"{field} is required")
    value = record[field]
    if not isinstance(value, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise SchemaError(f"{field} must be a list of integers")
    return value
