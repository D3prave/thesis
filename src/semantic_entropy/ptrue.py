"""P(True) supervised baseline (Kadavath et al. 2022; Farquhar et al. 2024).

P(True) is *not* an entropy. It is a supervised, in-context baseline: the model
samples several answers, is then shown the question, the brainstormed answers,
and one candidate ("most likely") answer, and is asked whether that candidate is
True. The probability the model assigns to "True" is its self-estimated
confidence; ``1 - P(True)`` is the uncertainty score used for AUROC (so that,
like the entropy estimators, a *higher* score means *more likely incorrect*).

This module owns the prompt construction and the score arithmetic — both pure
and unit-testable. The actual forward pass that turns a prompt into ``P(True)``
(reading the next-token probabilities of the "True"/"False" options) is injected
as a ``judge_fn`` callable, so the cluster job supplies a vLLM-backed judge while
tests supply a deterministic stub.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

#: A judge maps a fully-formed P(True) prompt to ``P("True")`` in ``[0, 1]``.
JudgeFn = Callable[[str], float]

_INSTRUCTION = (
    "Answer the following question. Consider the brainstormed answers, then "
    "decide whether the proposed answer is correct."
)


def most_likely_answer(record: Mapping[str, Any]) -> str:
    """Return the candidate ("most likely") answer for a record.

    Uses the most frequent sampled answer as a black-box proxy for the
    highest-probability generation (ties broken by first occurrence).
    """
    sampled = list(record.get("sampled_answers", []))
    if not sampled:
        raise ValueError("record has no sampled_answers")
    counts = Counter(sampled)
    best = max(sampled, key=lambda answer: (counts[answer], -sampled.index(answer)))
    return best


def format_ptrue_prompt(
    question: str,
    brainstormed_answers: Sequence[str],
    proposed_answer: str,
    few_shot_prefix: str = "",
) -> str:
    """Build the P(True) prompt for one record (authors' A/B format).

    Mirrors jlko/semantic_uncertainty ``calculate_p_true``: the question, the
    brainstormed answers followed by the proposed answer, then a True/False
    multiple choice. The model's probability on the "A" continuation is P(True).
    ``few_shot_prefix`` (from :func:`build_few_shot_prefix`) is prepended.
    """
    prefix = (few_shot_prefix.rstrip() + "\n") if few_shot_prefix else ""
    lines = [f"Question: {question.strip()}", "Brainstormed Answers: "]
    for answer in list(brainstormed_answers) + [proposed_answer]:
        lines.append(answer.strip())
    lines.append(f"Possible answer: {proposed_answer.strip()}")
    lines.append("Is the possible answer:")
    lines.append("A) True")
    lines.append("B) False")
    lines.append("The possible answer is:")
    return prefix + "\n".join(lines)


def build_few_shot_prefix(examples: Sequence[Mapping[str, Any]]) -> str:
    """Build a few-shot prefix from labelled training examples.

    Each example must provide ``question``, ``brainstormed_answers``,
    ``proposed_answer``, and a boolean ``is_correct``. The paper uses up to 20
    such in-context examples; the completion is " A" when correct, " B"
    otherwise (the token the judge reads at test time).
    """
    blocks = [_INSTRUCTION, ""]
    for example in examples:
        block = format_ptrue_prompt(
            str(example["question"]),
            list(example.get("brainstormed_answers", [])),
            str(example["proposed_answer"]),
            few_shot_prefix="",
        )
        verdict = " A" if bool(example["is_correct"]) else " B"
        blocks.append(f"{block}{verdict}")
        blocks.append("")
    return "\n".join(blocks).rstrip()


def ptrue_uncertainty(p_true: float) -> float:
    """Convert ``P(True)`` to an uncertainty score (higher = more uncertain).

    Returns ``1 - P(True)`` clamped to ``[0, 1]`` so it shares the AUROC
    orientation of the entropy estimators (positive class = incorrect).
    """
    if not 0.0 <= p_true <= 1.0:
        p_true = min(1.0, max(0.0, p_true))
    return 1.0 - p_true


def score_record_ptrue(
    record: Mapping[str, Any],
    judge_fn: JudgeFn,
    *,
    few_shot_prefix: str = "",
    question_field: str = "prompt",
) -> float:
    """Return the P(True) uncertainty score for a single record.

    The proposed answer is the low-temperature ``most_likely_answer`` when
    present (the paper's protocol), otherwise the modal sampled answer.
    """
    question = str(record.get(question_field, record.get("prompt", "")))
    brainstormed = list(record.get("sampled_answers", []))
    proposed = str(record.get("most_likely_answer") or most_likely_answer(record))
    prompt = format_ptrue_prompt(question, brainstormed, proposed, few_shot_prefix)
    p_true = float(judge_fn(prompt))
    return ptrue_uncertainty(p_true)
