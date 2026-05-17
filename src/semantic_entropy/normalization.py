"""Synthetic answer normalization helpers for smoke tests."""

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
    """Normalize one synthetic answer string."""

    if not isinstance(answer, str):
        raise TypeError("answer must be a string")

    normalized = re.sub(r"\s+", " ", answer.strip().lower())
    return NUMERIC_WORDS.get(normalized, normalized)


def normalize_answers(answers: Sequence[str]) -> list[str]:
    """Normalize a sequence of synthetic answer strings."""

    return [normalize_answer(answer) for answer in answers]
