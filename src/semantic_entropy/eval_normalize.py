"""Dataset-faithful answer normalization for TriviaQA and SVAMP evaluation.

This module provides the normalization functions used to compare model
outputs to reference answers during correctness evaluation and semantic
clustering. It replaces the minimal synthetic normalizer in
:mod:`semantic_entropy.normalization` for records whose ``dataset`` field
is ``"triviaqa"`` or ``"svamp"``.

**TriviaQA normalization** follows the official evaluation protocol from the
original TriviaQA paper and evaluation code (Joshi et al., 2017):

1. Lowercase.
2. Remove all ASCII punctuation.
3. Remove standalone articles ``a``, ``an``, ``the`` at word boundaries.
4. Collapse and strip whitespace.

This is the widely-used standard that makes string-exact matching robust to
surface differences in punctuation and article use, while preserving all
substantive content words.

**SVAMP normalization** is designed for short numeric answers produced by
grade-school math word problems. It:

1. Substitutes English number words (``zero``–``ninety``) with their digit
   equivalents.
2. Strips comma thousands-separators (``1,000`` → ``1000``).
3. Extracts the first numeric value (integer or decimal, optionally signed)
   from the string, so that verbose outputs like ``"the answer is 7 apples"``
   reduce to ``"7"``.
4. Formats whole-valued floats as integers (``7.0`` → ``"7"``).
5. Falls back to stripped lowercase for strings that contain no recognizable
   numeric value.

**Dispatcher** :func:`normalize_answer_for_dataset` routes to the appropriate
normalizer based on the ``dataset`` name. Unknown datasets fall back to a
generic normalization (lowercase + whitespace collapse) rather than raising,
so that synthetic smoke records continue to work unmodified.
"""

from __future__ import annotations

import re
import string


# ---------------------------------------------------------------------------
# Dataset name constants
# ---------------------------------------------------------------------------

TRIVIAQA_DATASET: str = "triviaqa"
SVAMP_DATASET: str = "svamp"

#: Set of dataset names that have a dedicated faithful normalizer.
BIO_DATASET: str = "bio"

#: Set of dataset names that have a dedicated faithful normalizer.
KNOWN_DATASETS: frozenset[str] = frozenset({TRIVIAQA_DATASET, SVAMP_DATASET, BIO_DATASET})


# ---------------------------------------------------------------------------
# TriviaQA normalizer
# ---------------------------------------------------------------------------

_PUNCTUATION: frozenset[str] = frozenset(string.punctuation)
_ARTICLE_RE: re.Pattern[str] = re.compile(r"\b(a|an|the)\b")


def normalize_triviaqa_answer(answer: str) -> str:
    """Normalize a TriviaQA answer using the official evaluation protocol.

    Applies four transformations in the order used by the original TriviaQA
    evaluation script (Joshi et al., 2017):

    1. **Lowercase** — ``"Paris"`` → ``"paris"``.
    2. **Remove punctuation** — all characters in ``string.punctuation`` are
       dropped; ``"Shakespeare's"`` → ``"shakespeares"``.
    3. **Remove articles** — standalone ``a``, ``an``, ``the`` at word
       boundaries are replaced with a space; ``"the Eiffel Tower"``
       → ``" eiffel tower"``.
    4. **Whitespace fix** — leading/trailing whitespace and internal runs of
       whitespace are normalized to single spaces; ``" eiffel tower"``
       → ``"eiffel tower"``.

    Args:
        answer: Raw answer string from a model or reference list.

    Returns:
        Normalized answer string. Empty input returns an empty string.
    """
    answer = answer.lower()
    answer = "".join(ch for ch in answer if ch not in _PUNCTUATION)
    answer = _ARTICLE_RE.sub(" ", answer)
    answer = " ".join(answer.split())
    return answer


# ---------------------------------------------------------------------------
# SVAMP normalizer
# ---------------------------------------------------------------------------

# English number words recognized as standalone answer strings.
# Covers zero through ninety (single-word forms only; compound forms such
# as "twenty-one" are not handled and fall through to the numeric extractor).
_NUMBER_WORDS: dict[str, str] = {
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
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "sixty": "60",
    "seventy": "70",
    "eighty": "80",
    "ninety": "90",
}

# Matches an optionally signed integer or decimal number.
# Examples: "7", "-7", "3.5", "-3.5".
# Does NOT match ".5" (no leading digit) to avoid false positives
# on punctuation.
_NUMBER_RE: re.Pattern[str] = re.compile(r"-?\d+(?:\.\d+)?")


def normalize_svamp_answer(answer: str) -> str:
    """Normalize a SVAMP answer to a canonical numeric string.

    The normalizer tries to extract a single numeric value from *answer*
    by applying, in order:

    1. **Number-word substitution** — if the entire stripped, lowercased
       string is a recognized English number word (``zero``–``ninety``),
       it is substituted with the corresponding digit string.
    2. **Comma removal** — commas between digits are stripped to collapse
       thousands separators: ``"1,000"`` → ``"1000"``.
    3. **Numeric extraction** — the first occurrence of a signed integer or
       decimal pattern is extracted.  This collapses verbose outputs such as
       ``"the answer is 7 apples"`` → ``"7"``.
    4. **Canonical formatting** — whole-valued floats are formatted without
       a decimal point (``7.0`` → ``"7"``); non-integer floats preserve their
       decimal (``3.5`` → ``"3.5"``).
    5. **Fallback** — if no numeric value is found, the stripped, lowercased
       string is returned unchanged.

    Args:
        answer: Raw answer string from a model or reference list.

    Returns:
        Canonical numeric string, or stripped lowercase fallback.
    """
    stripped = answer.strip()
    lower = stripped.lower()

    # 1. Whole-string number-word substitution.
    if lower in _NUMBER_WORDS:
        return _NUMBER_WORDS[lower]

    # 2. Remove commas used as thousands separators.
    cleaned = re.sub(r"(?<=\d),(?=\d)", "", stripped)

    # 3. Extract the first numeric value.
    match = _NUMBER_RE.search(cleaned)
    if match:
        num_str = match.group()
        try:
            val = float(num_str)
            # 4. Format: whole float → integer string, other → float string.
            int_val = int(val)
            if float(int_val) == val:
                return str(int_val)
            return str(val)
        except (ValueError, OverflowError):
            pass

    # 5. Fallback.
    return lower


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def normalize_answer_for_dataset(answer: str, dataset: str) -> str:
    """Normalize *answer* using the normalizer appropriate for *dataset*.

    Routes to :func:`normalize_triviaqa_answer` for ``"triviaqa"`` and to
    :func:`normalize_svamp_answer` for ``"svamp"``. Unknown dataset names
    receive a generic treatment (lowercase + whitespace collapse) rather than
    raising, so that synthetic smoke records and future datasets continue to
    work without modifications to this module.

    Args:
        answer: Raw answer string.
        dataset: Dataset identifier, e.g. ``"triviaqa"`` or ``"svamp"``.

    Returns:
        Normalized answer string.
    """
    if dataset == TRIVIAQA_DATASET:
        return normalize_triviaqa_answer(answer)
    if dataset == SVAMP_DATASET:
        return normalize_svamp_answer(answer)
    if dataset == BIO_DATASET:
        from semantic_entropy.normalization import normalize_freeform
        return normalize_freeform(answer)
    # Generic fallback: lowercase + whitespace normalization.
    return " ".join(answer.lower().split())


def normalize_answers_for_dataset(
    answers: list[str], dataset: str
) -> list[str]:
    """Apply :func:`normalize_answer_for_dataset` to every answer in *answers*.

    Args:
        answers: List of raw answer strings.
        dataset: Dataset identifier forwarded to
            :func:`normalize_answer_for_dataset`.

    Returns:
        List of normalized strings in the same order as *answers*.
    """
    return [normalize_answer_for_dataset(a, dataset) for a in answers]
