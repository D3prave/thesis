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


def _build_prompt_rewriter() -> tuple[
    Callable[[str], str], str, list[dict[str, str]]
]:
    """Return ``(rewrite, regime, demonstrations)`` from the environment.

    Shared by both factories below so the two entry points cannot drift: the
    plain :func:`make_model` path and the ``ModelFnWithStates`` path used when
    SEP features are collected must build byte-identical prompts, or the probe
    feature would be read from a different string than the entropy samples.
    """
    regime = _require_env("SE_REPLICATION_REGIME")
    if regime not in BRIEF_PROMPTS:
        raise ValueError(
            f"SE_REPLICATION_REGIME must be one of {sorted(BRIEF_PROMPTS)}, "
            f"got {regime!r}"
        )
    num_few_shot = int(
        os.environ.get("SE_REPLICATION_NUM_FEW_SHOT", DEFAULT_NUM_FEW_SHOT)
    )

    fixed = os.environ.get("SE_REPLICATION_FEWSHOT_JSON", "").strip()
    if fixed:
        # Preferred path. Their demonstrations are drawn once from
        # random_seed=10 and never vary; the multiple seeds in this study are
        # an addition for estimating sampling variance, so they must not also
        # move the prompt prefix. prepare_replication_data.py writes the same
        # five their generate_answers.py would select.
        raw = json.loads(Path(fixed).read_text(encoding="utf-8"))
        demonstrations = [
            {"question": str(item["question"]).strip(),
             "answer": str(item["answer"]).strip()}
            for item in raw[:num_few_shot]
        ]
        if len(demonstrations) != num_few_shot:
            raise ValueError(
                f"{fixed} holds {len(raw)} demonstrations, need {num_few_shot}"
            )
    else:
        # Legacy path: select from a pool with the run seed. Retained so older
        # launchers keep working, but it makes the prefix seed-dependent.
        seed = int(_require_env("SE_SEED"))
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

    def rewrite(prompt: str) -> str:
        context, question = context_index.get(prompt, ("", prompt))
        return build_prompt(
            question or prompt,
            demonstrations,
            regime=regime,
            context=context or None,
        )

    return rewrite, regime, demonstrations


def _require_system_prompt_disabled() -> None:
    """Fail unless ``SE_SYSTEM_PROMPT`` is explicitly disabled.

    The underlying adapters read this variable through
    ``adapters._env_system_prompt``, which treats an *absent* variable as
    "keep the factory default" -- and that default is the terse QA system
    message, "Answer the following question as briefly as possible. Give only
    the answer...". Leaving it unset would therefore wrap the faithful
    completion prompt in a chat template carrying a second, contradictory
    instruction, and the run would look plausible while being neither regime.

    Only ``""`` or ``"none"`` disable it. ``unset`` does not.
    """
    raw = os.environ.get("SE_SYSTEM_PROMPT")
    if raw is None or raw.strip().lower() not in {"", "none"}:
        raise ValueError(
            "SE_SYSTEM_PROMPT must be set to 'none' (or empty) for replication "
            "runs: the original sends plain text with no system message, and "
            "leaving this variable unset silently keeps the terse QA default. "
            f"Currently: {raw!r}"
        )


def _annotate(fn: Any, regime: str, demonstrations: list[dict[str, str]]) -> Any:
    fn.replication_regime = regime
    fn.replication_demonstrations = demonstrations
    fn.replication_brief = BRIEF_PROMPTS[regime]
    return fn


def make_model(
    inner_factory: Callable[..., Callable[[str, int], Any]] | None = None,
    **inner_kwargs: Any,
) -> Callable[[str, int], Any]:
    """Return a ``ModelFn`` that prompts the way the original does.

    Use with ``--model-module``. This path draws the sampled answers but
    collects no hidden-state feature; for runs that also train or score the
    probes, use :func:`make_model_with_states` instead.

    Args:
        inner_factory: Factory for the underlying model function. Defaults to
            :func:`semantic_entropy.adapters.make_model`. Injectable for tests.
        **inner_kwargs: Forwarded to *inner_factory*.

    Raises:
        ValueError: If required environment variables are missing or the regime
            is unknown.
    """
    rewrite, regime, demonstrations = _build_prompt_rewriter()
    _require_system_prompt_disabled()

    if inner_factory is None:
        from semantic_entropy.adapters import (
            make_phase1_model as _default_factory,
        )

        inner_factory = _default_factory
    inner = inner_factory(**inner_kwargs)

    def replication_model_fn(prompt: str, num_samples: int) -> Any:
        return inner(rewrite(prompt), num_samples)

    return _annotate(replication_model_fn, regime, demonstrations)


def make_model_with_states(
    inner_factory: Callable[..., Callable[[str, int], Any]] | None = None,
    **inner_kwargs: Any,
) -> Callable[[str, int], Any]:
    """Return a ``ModelFnWithStates`` that prompts the way the original does.

    Use with ``--model-with-states-module``, which is mutually exclusive with
    ``--model-module`` and is the only path that writes ``sep_hidden_state``.
    The prompt is rewritten identically to :func:`make_model`, so the probe
    feature is read from the same string the entropy samples were drawn under.

    ``SE_FEATURE_TEMPERATURE`` should be ``0.1`` for faithful runs: Farquhar et
    al. sample the most-likely response rather than decoding it greedily, and
    that same response supplies the graded answer, the P(True) prompt and the
    probe feature.

    Raises:
        ValueError: If required environment variables are missing or the regime
            is unknown.
    """
    rewrite, regime, demonstrations = _build_prompt_rewriter()
    _require_system_prompt_disabled()

    if inner_factory is None:
        from semantic_entropy.adapters import (
            make_phase1_model_with_states as _default_factory,
        )

        inner_factory = _default_factory
    inner = inner_factory(**inner_kwargs)

    def replication_states_fn(prompt: str, num_samples: int) -> Any:
        return inner(rewrite(prompt), num_samples)

    return _annotate(replication_states_fn, regime, demonstrations)
