"""Utilities for the semantic entropy thesis pilot."""

from semantic_entropy.normalization import normalize_answer, normalize_answers
from semantic_entropy.schema import SchemaError, validate_record
from semantic_entropy.scoring import (
    discrete_semantic_entropy,
    score_record,
    surface_entropy,
)

__all__ = [
    "SchemaError",
    "discrete_semantic_entropy",
    "normalize_answer",
    "normalize_answers",
    "score_record",
    "surface_entropy",
    "validate_record",
]
