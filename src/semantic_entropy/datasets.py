"""Dataset ingestion stubs for TriviaQA and SVAMP.

This module is the first real data-ingestion component of the Phase 1
pipeline. It reads tiny hand-crafted stub files from ``data/raw/`` that
mirror the formats of the two target datasets and yields :class:`PromptItem`
objects ready to be handed to the sampling harness.

**TriviaQA format** (open / unfiltered JSONL, one object per line)::

    {
      "QuestionId": "tc_0001",
      "Question": "What is the capital city of France?",
      "Answer": {
        "Value": "Paris",
        "Aliases": ["Paris", "Paris, France"]
      }
    }

``Aliases`` provides the full list of acceptable surface forms. When
``Aliases`` is absent or empty the loader falls back to the single
``Value`` string.

**SVAMP format** (math word-problem JSONL, one object per line)::

    {
      "ID": "svamp_0001",
      "Body": "Mia has 3 apples.",
      "Question": "She buys 4 more. How many apples does she have now?",
      "Formula": "3 + 4",
      "Answer": 7
    }

The prompt is formed by concatenating ``Body`` and ``Question`` with a
single space. The numeric ``Answer`` is converted to a string and stored
as the sole reference answer.

Both loaders are *stdlib-only* and do not call any ML model, NLI backend,
or network resource.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths to the committed stub files
# ---------------------------------------------------------------------------

_PACKAGE_DIR = Path(__file__).parent
_REPO_ROOT = _PACKAGE_DIR.parent.parent
TRIVIAQA_STUB_PATH: Path = _REPO_ROOT / "data" / "raw" / "triviaqa_stub.jsonl"
SVAMP_STUB_PATH: Path = _REPO_ROOT / "data" / "raw" / "svamp_stub.jsonl"


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class DatasetError(ValueError):
    """Raised when a dataset file is missing, unreadable, or malformed."""


@dataclass(frozen=True)
class PromptItem:
    """A single prompt extracted from a dataset before model sampling.

    Attributes:
        prompt_id: Stable identifier from the source dataset.
        dataset: Dataset name, e.g. ``"triviaqa"`` or ``"svamp"``.
        split: Dataset split, e.g. ``"validation"`` or ``"train"``.
        prompt: The full question text fed to the model.
        reference_answers: One or more acceptable answer strings.
    """

    prompt_id: str
    dataset: str
    split: str
    prompt: str
    reference_answers: list[str]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_triviaqa_records(
    path: Path | str = TRIVIAQA_STUB_PATH,
    *,
    split: str = "validation",
    dataset_label: str = "triviaqa",
) -> Iterator[PromptItem]:
    """Yield :class:`PromptItem` objects from a TriviaQA open-format JSONL.

    Each line must be a JSON object with at minimum the keys
    ``QuestionId``, ``Question``, and ``Answer``.  ``Answer`` must be a
    mapping with at least a ``"Value"`` key; ``"Aliases"`` is used when
    present and non-empty.

    Args:
        path: Path to the JSONL file. Defaults to the committed stub at
            ``data/raw/triviaqa_stub.jsonl``.
        split: Dataset split label written into every emitted
            :class:`PromptItem`.

    Yields:
        One :class:`PromptItem` per line in *path*.

    Raises:
        DatasetError: If *path* does not exist, cannot be read, or
            contains a structurally invalid record.
    """
    path = Path(path)
    for line_number, raw_line in _iter_lines(path):
        record = _parse_json(raw_line, line_number, path)
        _require_key(record, "QuestionId", line_number, path)
        _require_key(record, "Question", line_number, path)
        _require_key(record, "Answer", line_number, path)

        question_id = _require_string(record, "QuestionId", line_number, path)
        question = _require_string(record, "Question", line_number, path)
        answer = record["Answer"]

        if not isinstance(answer, dict):
            raise DatasetError(
                f"{path}:{line_number}: Answer must be a JSON object"
            )
        if "Value" not in answer:
            raise DatasetError(
                f"{path}:{line_number}: Answer.Value is required"
            )
        value = answer["Value"]
        if not isinstance(value, str):
            raise DatasetError(
                f"{path}:{line_number}: Answer.Value must be a string"
            )

        aliases: list[str] = []
        if "Aliases" in answer:
            raw_aliases = answer["Aliases"]
            if not isinstance(raw_aliases, list):
                raise DatasetError(
                    f"{path}:{line_number}: Answer.Aliases must be a list"
                )
            aliases = [a for a in raw_aliases if isinstance(a, str) and a]

        reference_answers = aliases if aliases else [value]

        yield PromptItem(
            prompt_id=question_id,
            dataset=dataset_label,
            split=split,
            prompt=question,
            reference_answers=reference_answers,
        )


def load_nqopen_records(
    path: Path | str,
    *,
    split: str = "validation",
) -> Iterator[PromptItem]:
    """Yield :class:`PromptItem` objects from an NQ-Open JSONL.

    NQ-Open answers are free-text short strings, like TriviaQA, so the same
    ``{QuestionId, Question, Answer:{Value, Aliases}}`` on-disk format and parser
    are reused; only the ``dataset`` label differs (``"nqopen"``), which routes
    answer normalization and the SQuAD-F1 accuracy rule appropriately.
    """
    yield from load_triviaqa_records(path, split=split, dataset_label="nqopen")


def load_svamp_records(
    path: Path | str = SVAMP_STUB_PATH,
    *,
    split: str = "validation",
) -> Iterator[PromptItem]:
    """Yield :class:`PromptItem` objects from a SVAMP-format JSONL.

    Each line must be a JSON object with the keys ``ID``, ``Body``,
    ``Question``, and ``Answer``. The full prompt is the concatenation
    ``Body + " " + Question``. The numeric ``Answer`` is converted to a
    string; integers are formatted without a decimal point (``"7"`` not
    ``"7.0"``).

    Args:
        path: Path to the JSONL file. Defaults to the committed stub at
            ``data/raw/svamp_stub.jsonl``.
        split: Dataset split label written into every emitted
            :class:`PromptItem`.

    Yields:
        One :class:`PromptItem` per line in *path*.

    Raises:
        DatasetError: If *path* does not exist, cannot be read, or
            contains a structurally invalid record.
    """
    path = Path(path)
    for line_number, raw_line in _iter_lines(path):
        record = _parse_json(raw_line, line_number, path)
        _require_key(record, "ID", line_number, path)
        _require_key(record, "Body", line_number, path)
        _require_key(record, "Question", line_number, path)
        _require_key(record, "Answer", line_number, path)

        item_id = str(record["ID"])
        body = _require_string(record, "Body", line_number, path)
        question = _require_string(record, "Question", line_number, path)

        raw_answer = record["Answer"]
        if not isinstance(raw_answer, (int, float)) or isinstance(raw_answer, bool):
            raise DatasetError(
                f"{path}:{line_number}: Answer must be a number"
            )
        # Format integers without a trailing ".0" so answers like 7 round-trip
        # through the normalizer cleanly.
        if isinstance(raw_answer, float) and raw_answer == int(raw_answer):
            answer_str = str(int(raw_answer))
        else:
            answer_str = str(raw_answer)

        prompt = f"{body} {question}"

        yield PromptItem(
            prompt_id=item_id,
            dataset="svamp",
            split=split,
            prompt=prompt,
            reference_answers=[answer_str],
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _iter_lines(path: Path) -> Iterator[tuple[int, str]]:
    """Yield ``(line_number, line)`` for non-empty lines in *path*."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                stripped = line.strip()
                if stripped:
                    yield line_number, stripped
    except FileNotFoundError:
        raise DatasetError(f"dataset file not found: {path}") from None
    except OSError as exc:
        raise DatasetError(f"cannot read dataset file {path}: {exc}") from exc


def _parse_json(
    line: str, line_number: int, path: Path
) -> dict:
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        raise DatasetError(
            f"{path}:{line_number}: invalid JSON: {exc.msg}"
        ) from exc
    if not isinstance(record, dict):
        raise DatasetError(
            f"{path}:{line_number}: record must be a JSON object"
        )
    return record


def _require_key(
    record: dict, key: str, line_number: int, path: Path
) -> None:
    if key not in record:
        raise DatasetError(
            f"{path}:{line_number}: required field '{key}' is missing"
        )


def _require_string(
    record: dict, key: str, line_number: int, path: Path
) -> str:
    value = record[key]
    if not isinstance(value, str):
        raise DatasetError(
            f"{path}:{line_number}: field '{key}' must be a string"
        )
    return value
