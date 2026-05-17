"""Utilities for the semantic entropy thesis pilot."""

from semantic_entropy.schema import SchemaError, validate_record
from semantic_entropy.scoring import (
    discrete_semantic_entropy,
    score_record,
    surface_entropy,
)

__all__ = [
    "SchemaError",
    "discrete_semantic_entropy",
    "score_record",
    "surface_entropy",
    "validate_record",
]
