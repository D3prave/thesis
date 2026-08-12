"""Model adapter that reproduces the Farquhar et al. (2024) prompt format.

The pipeline calls a ``ModelFn`` with the record's prompt string and a sample
count. This adapter sits in front of the ordinary HuggingFace adapter and
rewrites that prompt into the layout their released ``generate_answers.py``
builds: one brief instruction, five worked ``Question:``/``Answer:``
demonstrations drawn once per run, and then the query, all as plain text with
no chat template and no system message.

Configured entirely through the environment so a launcher can select the
regime without code changes:

``SE_REPLICATION_REGIME``
    ``chat`` for the sentence-length setting of the main text, ``default`` for
    the short-phrase setting of Supplementary Note 7. Required.
``SE_REPLICATION_FEWSHOT_DATA``
    JSONL of *training* records from which demonstrations are drawn. Required.
``SE_REPLICATION_EVAL_DATA``
    JSONL of the evaluation records. Optional; supplying it lets SVAMP
    reconstruct its separate ``Context:`` line, which the original uses because
    it forces ``use_context=True`` for that dataset alone.
``SE_SEED``
    Run seed, used to pick demonstrations. Required.
``SE_REPLICATION_NUM_FEW_SHOT``
    Defaults to 5, their released default.

Everything else — model id, revisions, decoding — is read by the underlying
adapter from its own environment variables, unchanged.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from semantic_entropy.replication_prompts import (
    BRIEF_PROMPTS,
    DEFAULT_NUM_FEW_SHOT,
    build_prompt,
    demonstrations_from_items,
    select_demonstrations,
)


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} must be set for the replication adapter")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    if not records:
        raise ValueError(f"{path} contains no records")
    return records


def _as_candidate(record: dict[str, Any]) -> dict[str, Any]:
    """Normalise a prepared JSONL record into question/answer form."""
    question = record.get("Question") or record.get("question") or ""
    body = record.get("Body") or record.get("context") or ""
    answer_block = record.get("Answer") or record.get("answer") or {}
    if isinstance(answer_block, dict):
        aliases = answer_block.get("Aliases") or answer_block.get("aliases") or []
        value = answer_block.get("Value") or answer_block.get("value") or ""
        answers = [value, *aliases] if value else list(aliases)
    elif isinstance(answer_block, (list, tuple)):
        answers = list(answer_block)
    else:
        answers = [answer_block] if answer_block else []
    answers = [str(a).strip() for a in answers if str(a).strip()]
    return {
        "question": str(question).strip(),
        "context": str(body).strip(),
        "reference_answers": answers,
    }


def build_context_index(records: Sequence[dict[str, Any]]) -> dict[str, tuple[str, str]]:
    """Map the concatenated prompt back to its ``(context, question)`` parts.

    SVAMP is the only dataset that needs this. Its prepared records keep
    ``Body`` and ``Question`` separate, while the pipeline passes the
    concatenation, so the split is recovered by lookup rather than by guessing
    a delimiter.
    """
    index: dict[str, tuple[str, str]] = {}
    for record in records:
        candidate = _as_candidate(record)
        question = candidate["question"]
        context = candidate["context"]
        if not question:
            continue
        joined = f"{context} {question}".strip() if context else question
        index[joined] = (context, question)
    return index


def make_model(
    inner_factory: Callable[..., Callable[[str, int], Any]] | None = None,
    **inner_kwargs: Any,
) -> Callable[[str, int], Any]:
    """Return a ``ModelFn`` that prompts the way the original does.

    Args:
        inner_factory: Factory for the underlying model function. Defaults to
            :func:`semantic_entropy.adapters.make_model`. Injectable for tests.
        **inner_kwargs: Forwarded to *inner_factory*.

    Raises:
        ValueError: If required environment variables are missing or the regime
            is unknown.
    """
    regime = _require_env("SE_REPLICATION_REGIME")
    if regime not in BRIEF_PROMPTS:
        raise ValueError(
            f"SE_REPLICATION_REGIME must be one of {sorted(BRIEF_PROMPTS)}, "
            f"got {regime!r}"
        )
    seed = int(_require_env("SE_SEED"))
    num_few_shot = int(
        os.environ.get("SE_REPLICATION_NUM_FEW_SHOT", DEFAULT_NUM_FEW_SHOT)
    )

    train_records = _load_jsonl(Path(_require_env("SE_REPLICATION_FEWSHOT_DATA")))
    candidates = [_as_candidate(record) for record in train_records]
    few_shot_indices, _, _ = select_demonstrations(
        candidates, seed=seed, num_few_shot=num_few_shot, num_p_true=num_few_shot
    )
    demonstrations = demonstrations_from_items(candidates, few_shot_indices)

    eval_path = os.environ.get("SE_REPLICATION_EVAL_DATA", "").strip()
    context_index: dict[str, tuple[str, str]] = {}
    if eval_path:
        context_index = build_context_index(_load_jsonl(Path(eval_path)))

    if inner_factory is None:
        from semantic_entropy.adapters import make_model as _default_factory

        inner_factory = _default_factory
    # The original sends plain text, so any system message must be suppressed.
    inner_kwargs.setdefault("system_prompt", None)
    inner = inner_factory(**inner_kwargs)

    def replication_model_fn(prompt: str, num_samples: int) -> Any:
        context, question = context_index.get(prompt, ("", prompt))
        full_prompt = build_prompt(
            question or prompt,
            demonstrations,
            regime=regime,
            context=context or None,
        )
        return inner(full_prompt, num_samples)

    replication_model_fn.replication_regime = regime  # type: ignore[attr-defined]
    replication_model_fn.replication_demonstrations = demonstrations  # type: ignore[attr-defined]
    replication_model_fn.replication_brief = BRIEF_PROMPTS[regime]  # type: ignore[attr-defined]
    return replication_model_fn
