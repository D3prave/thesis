"""Public interface for the semantic-entropy thesis pilot.

The package implements the Phase 1 discrete semantic-entropy baseline of the
thesis: synthetic answer normalization, structural validation of the shared
JSONL record schema, and two uncertainty estimators -- surface-form entropy
and discrete semantic entropy -- in a deliberately small, dependency-free
form. Probability-weighted semantic entropy and the Phase 2 extensions
(SEP, KLE, Semantic Energy, Semantic Volume, adaptive Bayesian SE) are out
of scope and are not re-exported here.
"""

from semantic_entropy.cluster_check import (
    check_cluster_consistency,
    check_jsonl_cluster_consistency,
)
from semantic_entropy.normalization import normalize_answer, normalize_answers
from semantic_entropy.schema import SchemaError, validate_record
from semantic_entropy.scoring import (
    discrete_semantic_entropy,
    score_record,
    surface_entropy,
)

__all__ = [
    "SchemaError",
    "check_cluster_consistency",
    "check_jsonl_cluster_consistency",
    "discrete_semantic_entropy",
    "normalize_answer",
    "normalize_answers",
    "score_record",
    "surface_entropy",
    "validate_record",
]
