"""Synthetic answer normalization for Phase 1 smoke tests.

This module provides a deliberately minimal canonicalization step intended
for synthetic fixtures and end-to-end plumbing checks. It is *not* the
dataset-faithful normalizer that TriviaQA, SVAMP, and the other Phase 1/2
datasets require: it performs only case folding, internal-whitespace
collapsing, and a closed-vocabulary substitution of the English number words
``zero``--``ten`` to their digit form. Punctuation, articles, alias lookup,
and the wider numeric vocabulary needed by real evaluation are intentionally
out of scope and are supplied by the dataset-specific normalizers in
``semantic_entropy.eval_normalize``.

Keeping the synthetic normalizer in a separate module makes the boundary
between smoke-test plumbing and real evaluation logic explicit and prevents
inadvertent reuse of the simplified rules in production scoring.
"""

from __future__ import annotations

import re
from collections.abc import Sequence


NUMERIC_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


def normalize_answer(answer: str) -> str:
    """Apply the synthetic canonicalization rules to a single answer string.

    The transformation is: strip leading and trailing whitespace, lower-case,
    collapse runs of internal whitespace to a single space, and -- if the
    resulting string matches one of the closed-vocabulary English number
    words ``zero``--``ten`` exactly -- substitute its digit form. The
    substitution is intentionally whole-string and case-folded, so phrases
    such as ``"seven apples"`` are left unchanged. A non-string input raises
    :class:`TypeError`, mirroring the strict-failure stance taken by the
    rest of the pipeline.
    """

    if not isinstance(answer, str):
        raise TypeError("answer must be a string")

    normalized = re.sub(r"\s+", " ", answer.strip().lower())
    return NUMERIC_WORDS.get(normalized, normalized)


def normalize_answers(answers: Sequence[str]) -> list[str]:
    """Apply :func:`normalize_answer` element-wise to a sequence of answers."""

    return [normalize_answer(answer) for answer in answers]


# ---------------------------------------------------------------------------
# Free-form normalizer for long-form generation (bio task)
# ---------------------------------------------------------------------------


def normalize_freeform(answer: str) -> str:
    """Minimal normalization for free-form / long-form answers.

    Designed for multi-sentence paragraph responses where dataset-specific
    normalization (TriviaQA punctuation stripping, SVAMP numeric extraction)
    is inappropriate. Applies only:

    1. Strip leading/trailing whitespace.
    2. Lowercase.
    3. Collapse runs of internal whitespace to a single space.
    4. Strip leading and trailing ASCII punctuation (periods, commas,
       colons, etc.) that frequently appear at paragraph boundaries.

    The output retains all substantive content, making it suitable as the
    ``normalized_answers`` input to exact-match clustering (which will
    produce near-degenerate clusters — one per sample — as expected for
    long-form text, demonstrating exact-match collapse on long-form tasks).

    Args:
        answer: Raw model output string.

    Returns:
        Normalized string. Empty input returns an empty string.
    """
    if not isinstance(answer, str):
        raise TypeError("answer must be a string")
    # Lowercase + collapse whitespace.
    result = re.sub(r"\s+", " ", answer.strip().lower())
    # Strip boundary punctuation.
    result = result.strip(".,;:!?\"'")
    return result


def normalize_freeform_answers(answers: Sequence[str]) -> list[str]:
    """Apply :func:`normalize_freeform` element-wise."""
    return [normalize_freeform(a) for a in answers]
