"""LLM-as-judge for bidirectional semantic-entailment clustering.

Provides :class:`BatchedLlmJudge`, a vLLM-backed judge that takes
``(question, premise, hypothesis)`` triples and returns one of the NLI
label constants from :mod:`semantic_entropy.clustering`. Designed to
substitute for the cross-encoder NLI judge in the Phase~2 clustering
step on the same ordered-pair contract used by
:func:`semantic_entropy.clustering.nli_cluster`.

Each ordered pair is judged independently with a Yes/No prompt that
asks whether the *premise* entails the *hypothesis* as a correct
response to the *question*. Bidirectional entailment is enforced by
the union-find pass in :func:`nli_cluster` (here applied per record
by the companion script ``scripts/posthoc_llm_judge_jsonl.py``).

Why this exists
---------------
The Phase~2 posthoc reruns under
:mod:`scripts.posthoc_recluster_jsonl` only support cross-encoder NLI
models (DeBERTa-v3-base, DeBERTa-v3-large). To test whether the
remaining gap to Farquhar et al.~\\cite{semantic_entropy_nature_2024}
on TriviaQA is attributable to judge strength, the same clustering
step is rerun with an instruction-tuned open judge of capability
comparable to or greater than the GPT-3.5 judge used in the original
paper (Qwen2.5-72B-Instruct by default).
"""

from __future__ import annotations

from collections.abc import Iterable

from semantic_entropy.clustering import NLI_ENTAILMENT, NLI_NEUTRAL


#: Type alias for a single judge task. ``(question, premise, hypothesis)``.
JudgeTriple = tuple[str, str, str]


DEFAULT_SYSTEM_PROMPT = (
    "You are a careful semantic judge. You are given a question and two "
    "candidate answers. Your job is to decide whether the first answer "
    "entails the second as a correct response to the question."
)


DEFAULT_USER_TEMPLATE = (
    "Question: {question}\n"
    "Answer A: {premise}\n"
    "Answer B: {hypothesis}\n\n"
    "Does Answer A entail Answer B? In other words, in the context of the "
    "question, if Answer A is a correct response, must Answer B also be a "
    "correct response with the same meaning?\n"
    "Reply with exactly one word: \"yes\" or \"no\"."
)

# ---------------------------------------------------------------------------
# Equivalence mode — for long-form / bio clustering
# ---------------------------------------------------------------------------

EQUIVALENCE_SYSTEM_PROMPT = (
    "You are a careful semantic judge. You are given a question and two "
    "candidate responses. Your job is to decide whether the two responses "
    "convey the same set of factual claims."
)

EQUIVALENCE_USER_TEMPLATE = (
    "Question: {question}\n"
    "Response A: {premise}\n"
    "Response B: {hypothesis}\n\n"
    "Do A and B convey the same set of factual claims about the subject?\n"
    "Treat A and B as equivalent if every specific factual claim made by A is "
    "also made (or trivially implied) by B, AND every specific factual claim "
    "made by B is also made (or trivially implied) by A. Stylistic differences, "
    "ordering, and verbosity do NOT affect equivalence. A claim only present in "
    "one of the two responses breaks equivalence.\n"
    "Answer with exactly one token: YES or NO."
)


def parse_judgment(raw: str) -> str:
    """Map a raw LLM response to :data:`NLI_ENTAILMENT` or :data:`NLI_NEUTRAL`.

    Robust to leading whitespace, quotes, and capitalization. Any
    response that does not begin with the token ``yes`` is treated as
    non-entailment, on the principle that a properly calibrated judge
    should commit explicitly to ``yes`` for entailment.

    Args:
        raw: Free-text response from the judge model.

    Returns:
        Either :data:`NLI_ENTAILMENT` or :data:`NLI_NEUTRAL`. The
        ``contradiction`` label is intentionally never returned --- this
        judge only distinguishes entailment from non-entailment, since
        only entailment is acted upon by :func:`nli_cluster`.
    """
    lowered = raw.strip().lower()
    # Tolerate punctuation/quotes around the yes/no token.
    lowered = lowered.lstrip("\"' .:,-*")
    if lowered.startswith("yes"):
        return NLI_ENTAILMENT
    return NLI_NEUTRAL


class BatchedLlmJudge:
    """vLLM-backed bidirectional NLI judge over ``(question, premise, hypothesis)`` triples.

    The judge loads a single instruction-tuned model with
    ``vllm.LLM`` and runs short-completion ``yes``/``no`` queries in
    one batched ``generate`` call. ``tensor_parallel_size`` should be
    set to the number of GPUs in the Slurm allocation.

    Example
    -------
    >>> judge = BatchedLlmJudge(model_name="Qwen/Qwen2.5-72B-Instruct",
    ...                         tensor_parallel_size=4)  # doctest: +SKIP
    >>> labels = judge.judge([                           # doctest: +SKIP
    ...     ("What is the capital of France?", "paris", "paris, france"),
    ... ])                                               # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        model_name: str,
        tensor_parallel_size: int = 4,
        gpu_memory_utilization: float = 0.90,
        dtype: str = "auto",
        max_model_len: int | None = None,
        enforce_eager: bool = True,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        user_template: str = DEFAULT_USER_TEMPLATE,
        max_tokens: int = 4,
        revision: str | None = None,
    ) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:  # pragma: no cover - deferred dep
            raise ImportError(
                "BatchedLlmJudge requires vllm. Install with: "
                "pip install -e '.[vllm]'"
            ) from exc

        self._llm = LLM(
            model=model_name,
            revision=revision,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
        )
        self._sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
        )
        self._system_prompt = system_prompt
        self._user_template = user_template
        self._tokenizer = self._llm.get_tokenizer()
        self.model_name = model_name
        print(
            f"loaded LLM judge {model_name} "
            f"(tp={tensor_parallel_size}, max_tokens={max_tokens})",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def judge(self, triples: list[JudgeTriple]) -> dict[JudgeTriple, str]:
        """Return entailment labels for each ``(question, premise, hypothesis)`` triple.

        Args:
            triples: List of judge tasks. Duplicates are not deduplicated
                here --- the caller is expected to deduplicate upstream
                (see ``scripts/posthoc_llm_judge_jsonl.py``).

        Returns:
            Mapping from each input triple to one of
            :data:`NLI_ENTAILMENT` or :data:`NLI_NEUTRAL`.
        """
        if not triples:
            return {}
        prompts = [self._format_prompt(t) for t in triples]
        outputs = self._llm.generate(prompts, self._sampling_params)
        labels: dict[JudgeTriple, str] = {}
        for triple, out in zip(triples, outputs):
            raw = out.outputs[0].text if out.outputs else ""
            labels[triple] = parse_judgment(raw)
        return labels

    def judge_in_batches(
        self,
        triples: list[JudgeTriple],
        *,
        batch_size: int = 1024,
    ) -> dict[JudgeTriple, str]:
        """Same as :meth:`judge`, but yields progress every ``batch_size`` triples.

        vLLM internally batches across a single ``generate`` call, so
        the chunking here is for progress reporting rather than for
        memory management.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        labels: dict[JudgeTriple, str] = {}
        total = len(triples)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            chunk = triples[start:end]
            labels.update(self.judge(chunk))
            print(f"  judged {end}/{total} triples", flush=True)
        return labels

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _format_prompt(self, triple: JudgeTriple) -> str:
        question, premise, hypothesis = triple
        user_msg = self._user_template.format(
            question=question, premise=premise, hypothesis=hypothesis
        )
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_msg},
        ]
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def enumerate_ordered_pairs(
    answers: list[str],
) -> Iterable[tuple[int, int, str, str]]:
    """Yield ``(i, j, answers[i], answers[j])`` for every ordered pair ``i != j``.

    Helper used by the posthoc script to collect bidirectional pair
    tasks (each unordered pair contributes two tasks, one in each
    direction). Identical strings are still emitted as separate tasks
    so the caller can decide whether to short-circuit them.
    """
    n = len(answers)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            yield i, j, answers[i], answers[j]


__all__ = [
    "BatchedLlmJudge",
    "JudgeTriple",
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_USER_TEMPLATE",
    "EQUIVALENCE_SYSTEM_PROMPT",
    "EQUIVALENCE_USER_TEMPLATE",
    "parse_judgment",
    "enumerate_ordered_pairs",
]
