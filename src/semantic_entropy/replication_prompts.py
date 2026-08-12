"""Prompt construction faithful to Farquhar et al. (2024).

The released code at ``github.com/jlko/semantic_uncertainty`` builds a plain
text completion prompt: one brief instruction, then five worked
``Question:``/``Answer:`` demonstrations, then the query. It does **not** use a
chat template and does not send a system message, even for instruction-tuned
models.

Two brief instructions select the answer-length regime. They are reproduced
verbatim from ``uncertainty/utils/utils.py``::

    BRIEF_PROMPTS = {
        'default': "Answer the following question as briefly as possible.\\n",
        'chat': 'Answer the following question in a single brief but complete sentence.\\n'}

``default`` produces the short-phrase setting of Supplementary Note 7 (mean
answer length 15.9 +/- 21.8 characters); ``chat`` produces the sentence-length
setting of the main text (95.6 +/- 69.6 characters). Those two figures are the
only external check that a run used the prompt it claims, which is what
:func:`assert_answer_length` exists for: an identity gate cannot catch a run
that is well-formed but was generated under the wrong instruction.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence

#: Verbatim from the released ``uncertainty/utils/utils.py``.
BRIEF_PROMPTS: dict[str, str] = {
    "default": "Answer the following question as briefly as possible.\n",
    "chat": "Answer the following question in a single brief but complete sentence.\n",
}

#: Reported mean and standard deviation of answer length, in characters,
#: for each regime (Supplementary Note 7).
EXPECTED_ANSWER_CHARS: dict[str, tuple[float, float]] = {
    "default": (15.9, 21.8),
    "chat": (95.6, 69.6),
}

#: Released defaults that this study matches.
DEFAULT_NUM_FEW_SHOT = 5
DEFAULT_MAX_NEW_TOKENS = 50
DEFAULT_NUM_GENERATIONS = 10
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 50
DEFAULT_MOST_LIKELY_TEMPERATURE = 0.1
DEFAULT_P_TRUE_NUM_FEW_SHOT = 20


def make_prompt(
    question: str,
    answer: str | None = None,
    *,
    context: str | None = None,
    brief: str = "",
    brief_always: bool = False,
) -> str:
    """Format one demonstration or query, following their ``get_make_prompt``.

    Args:
        question: The question text.
        answer: The answer for a demonstration. ``None`` or empty leaves the
            prompt open at ``"Answer:"`` for the model to continue.
        context: Passage text. The released default is ``use_context=False``
            and no dataset in this study supplies one.
        brief: Brief instruction, included only when *brief_always*.
        brief_always: Their ``--brief_always``, default ``False``.

    Returns:
        The formatted fragment. Demonstrations end with a blank line; an open
        query ends at ``"Answer:"`` with no trailing newline.
    """
    prompt = brief if brief_always else ""
    if context:
        prompt += f"Context: {context}\n"
    prompt += f"Question: {question}\n"
    if answer:
        prompt += f"Answer: {answer}\n\n"
    else:
        prompt += "Answer:"
    return prompt


def build_prompt(
    question: str,
    demonstrations: Sequence[Mapping[str, str]] = (),
    *,
    regime: str = "chat",
    brief_always: bool = False,
    context: str | None = None,
) -> str:
    """Assemble a full completion prompt for one query.

    Mirrors ``construct_fewshot_prompt_from_indices``: when *brief_always* is
    false the brief instruction is emitted once, at the top, and not repeated
    before each demonstration.

    Args:
        question: The query.
        demonstrations: Few-shot examples, each a mapping with ``question`` and
            ``answer`` keys. Five is the released default.
        regime: ``"chat"`` for sentence-length, ``"default"`` for short-phrase.
        brief_always: Repeat the brief instruction before every demonstration.
        context: Optional passage; unused in this study.

    Raises:
        ValueError: If *regime* is not a known brief prompt, or a demonstration
            is missing a question or an answer.
    """
    if regime not in BRIEF_PROMPTS:
        raise ValueError(
            f"unknown regime {regime!r}; expected one of {sorted(BRIEF_PROMPTS)}"
        )
    brief = BRIEF_PROMPTS[regime]

    prompt = "" if brief_always else brief
    for index, demo in enumerate(demonstrations):
        demo_question = str(demo.get("question", "")).strip()
        demo_answer = str(demo.get("answer", "")).strip()
        if not demo_question or not demo_answer:
            raise ValueError(
                f"demonstration {index} needs a non-empty question and answer"
            )
        prompt += make_prompt(
            demo_question,
            demo_answer,
            brief=brief,
            brief_always=brief_always,
        )
    prompt += make_prompt(
        question, None, context=context, brief=brief, brief_always=brief_always
    )
    return prompt


def select_demonstrations(
    candidates: Sequence[Mapping[str, object]],
    *,
    seed: int,
    num_few_shot: int = DEFAULT_NUM_FEW_SHOT,
    num_p_true: int = DEFAULT_P_TRUE_NUM_FEW_SHOT,
) -> tuple[list[int], list[int], list[int]]:
    """Choose few-shot and P(True) demonstrations the way the original does.

    Their ``generate_answers.py`` seeds the global random module once with the
    run seed and then draws twice, in this order::

        prompt_indices = random.sample(answerable_indices, args.num_few_shot)
        remaining_answerable = list(set(answerable_indices) - set(prompt_indices))
        p_true_indices = random.sample(answerable_indices, args.p_true_num_fewshot)
        remaining_answerable = list(set(remaining_answerable) - set(p_true_indices))

    Order matters, because the second draw consumes the same stream as the
    first. Both draws are taken from *answerable* examples only — those with at
    least one reference answer — and both sets are then removed from the pool
    the training split is evaluated over.

    Note the original draws ``p_true_indices`` from the full answerable pool
    rather than from what remains, so the two sets can overlap. That is
    reproduced.

    Args:
        candidates: Training examples. An example counts as answerable when its
            ``reference_answers`` (or ``answers``) entry is non-empty.
        seed: Run seed, matching ``--random_seed``.
        num_few_shot: Demonstrations for the generation prompt.
        num_p_true: Demonstrations for the P(True) prompt.

    Returns:
        ``(few_shot_indices, p_true_indices, remaining_indices)`` as positions
        into *candidates*.

    Raises:
        ValueError: If too few answerable examples exist for either draw.
    """
    import random as _random

    answerable = [
        index
        for index, item in enumerate(candidates)
        if _has_reference(item)
    ]
    if len(answerable) < num_few_shot:
        raise ValueError(
            f"need {num_few_shot} answerable examples, found {len(answerable)}"
        )
    if len(answerable) < num_p_true:
        raise ValueError(
            f"need {num_p_true} answerable examples for P(True), "
            f"found {len(answerable)}"
        )

    rng = _random.Random(seed)
    few_shot = rng.sample(answerable, num_few_shot)
    remaining = set(answerable) - set(few_shot)
    p_true = rng.sample(answerable, num_p_true)
    remaining -= set(p_true)

    return few_shot, p_true, sorted(remaining)


def _has_reference(item: Mapping[str, object]) -> bool:
    for key in ("reference_answers", "answers"):
        value = item.get(key)
        if isinstance(value, Mapping):
            value = value.get("text")
        if isinstance(value, (list, tuple)) and any(
            str(entry).strip() for entry in value
        ):
            return True
        if isinstance(value, str) and value.strip():
            return True
    return False


def demonstrations_from_items(
    candidates: Sequence[Mapping[str, object]],
    indices: Sequence[int],
) -> list[dict[str, str]]:
    """Turn selected training examples into ``{question, answer}`` pairs.

    The original takes the *first* reference answer as the demonstration
    answer (``example["answers"]["text"][0]``).
    """
    demonstrations: list[dict[str, str]] = []
    for index in indices:
        item = candidates[index]
        question = str(item.get("question") or item.get("prompt") or "").strip()
        answers = item.get("reference_answers") or item.get("answers") or []
        if isinstance(answers, Mapping):
            answers = answers.get("text", [])
        if isinstance(answers, str):
            answers = [answers]
        answer = str(answers[0]).strip() if answers else ""
        if not question or not answer:
            raise ValueError(f"candidate {index} has no usable question/answer")
        demonstrations.append({"question": question, "answer": answer})
    return demonstrations


def assert_answer_length(
    answers: Sequence[str],
    regime: str,
    *,
    tolerance_sds: float = 1.0,
) -> float:
    """Fail loudly when generations do not match the regime they claim.

    Wrapping a launcher inherits its whole environment, so a run can silently
    keep an earlier brief instruction and still finish, producing well-formed
    output for the wrong setting. Comparing mean answer length against the
    published figure is the cheapest external check that this did not happen.

    Args:
        answers: Generated answers, one per sample.
        regime: The regime the run claims to be in.
        tolerance_sds: How many published standard deviations of slack to allow
            around the published mean.

    Returns:
        The observed mean answer length in characters.

    Raises:
        ValueError: If *answers* is empty or *regime* is unknown.
        AssertionError: If the observed mean falls outside the tolerance.
    """
    if regime not in EXPECTED_ANSWER_CHARS:
        raise ValueError(f"unknown regime {regime!r}")
    if not answers:
        raise ValueError("answers must not be empty")

    observed = statistics.fmean(len(answer) for answer in answers)
    expected, sd = EXPECTED_ANSWER_CHARS[regime]
    low = expected - tolerance_sds * sd
    high = expected + tolerance_sds * sd
    if not low <= observed <= high:
        other = "default" if regime == "chat" else "chat"
        other_expected = EXPECTED_ANSWER_CHARS[other][0]
        hint = (
            f" This is closer to the {other!r} regime (~{other_expected:.0f} "
            "chars); check that the brief instruction reached the model."
            if abs(observed - other_expected) < abs(observed - expected)
            else ""
        )
        raise AssertionError(
            f"mean answer length {observed:.1f} chars is outside "
            f"[{low:.1f}, {high:.1f}] for regime {regime!r} "
            f"(published {expected} +/- {sd})." + hint
        )
    return observed
