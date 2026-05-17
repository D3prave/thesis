"""Model adapter interface for the semantic-entropy sampling pipeline.

This module defines the :data:`ModelFn` contract shared across all stages of
the thesis and provides two concrete implementations:

* :class:`SyntheticModel` — deterministic stub used in unit tests and local
  smoke runs. Requires no external dependencies.
* :func:`make_hf_model` — factory that wraps a HuggingFace
  ``transformers.pipeline`` as a :data:`ModelFn`. Requires ``torch`` and
  ``transformers``; raises :class:`ImportError` with an install hint if those
  packages are absent (deferred to Stage 5).
* :func:`make_vllm_model` — factory that wraps ``vllm.LLM`` for batched
  inference on a single GPU. Requires ``vllm``; raises :class:`ImportError`
  if absent (deferred to Stage 5).

**How to swap in a real model for Stage 5:**

1. Install the heavy dependencies::

       pip install torch transformers accelerate  # HuggingFace path
       pip install vllm                           # vLLM path

2. Create a :data:`ModelFn` via one of the factories and pass it to
   :func:`~semantic_entropy.harness.run_pipeline`::

       from semantic_entropy.models import make_hf_model
       model_fn = make_hf_model(
           "mistralai/Mistral-7B-Instruct-v0.2",
           temperature=0.7, top_p=0.95, max_new_tokens=64,
       )
       run_pipeline(items, config, model_fn, output_path)

3. For an NLI entailment function pass a compatible :data:`NliFn` to
   ``run_pipeline(..., entailment_fn=...)``::

       from semantic_entropy.models import make_nli_fn
       entailment_fn = make_nli_fn("cross-encoder/nli-deberta-v3-base")
       run_pipeline(items, config, model_fn, output_path,
                    entailment_fn=entailment_fn)
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence

# Re-export the canonical type alias so callers can import from one place.
from semantic_entropy.clustering import NLI_ENTAILMENT, NLI_NEUTRAL, NliFn

#: Type alias for the sampling model function.
#:
#: A ``ModelFn`` takes a prompt string and an integer sample count ``n`` and
#: returns a list of exactly ``n`` generated answer strings.  The strings
#: should be the *raw* model output before any normalization; normalization
#: is applied by :func:`~semantic_entropy.harness.sample_record`.
ModelFn = Callable[[str, int], list[str]]


# ---------------------------------------------------------------------------
# Synthetic stub (no external deps)
# ---------------------------------------------------------------------------


class SyntheticModel:
    """Deterministic stub model for unit tests and smoke runs.

    Given a fixed answer pool, :meth:`__call__` samples ``n`` answers with
    replacement in a reproducible round-robin fashion so tests can predict
    exact outputs without setting global random state.

    Args:
        answers: Pool of answer strings to sample from.  Must be non-empty.
        seed: Optional integer seed; if provided the pool is shuffled once
            on construction so that different seeds yield different orderings.

    Example::

        model = SyntheticModel(["paris", "lyon", "paris"])
        assert model("any prompt", 2) == ["paris", "lyon"]
    """

    def __init__(self, answers: Sequence[str], seed: int | None = None) -> None:
        if not answers:
            raise ValueError("answers pool must not be empty")
        pool = list(answers)
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(pool)
        self._pool = pool

    def __call__(self, prompt: str, n: int) -> list[str]:  # noqa: ARG002
        """Return ``n`` answers sampled round-robin from the pool."""
        return [self._pool[i % len(self._pool)] for i in range(n)]


# ---------------------------------------------------------------------------
# HuggingFace model factory (deferred to Stage 5)
# ---------------------------------------------------------------------------


def make_hf_model(
    model_name: str,
    *,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_new_tokens: int = 64,
    device_map: str = "auto",
    torch_dtype: str = "auto",
) -> ModelFn:
    """Return a :data:`ModelFn` backed by a HuggingFace text-generation pipeline.

    Requires ``torch`` and ``transformers``.  Raises :class:`ImportError` with
    an install hint if either package is missing.

    Args:
        model_name: HuggingFace Hub model ID, e.g.
            ``"mistralai/Mistral-7B-Instruct-v0.2"``.
        temperature: Sampling temperature.
        top_p: Nucleus-sampling probability threshold.
        max_new_tokens: Maximum number of tokens to generate per answer.
        device_map: Passed directly to the HF pipeline (``"auto"`` works for
            single-GPU and multi-GPU setups).
        torch_dtype: Passed directly to the HF pipeline (``"auto"`` selects
            bfloat16 on supported hardware).

    Returns:
        A :data:`ModelFn` whose ``__call__`` signature is
        ``(prompt: str, n: int) -> list[str]``.

    Raises:
        ImportError: If ``torch`` or ``transformers`` are not installed.
    """
    try:
        import torch  # noqa: F401
        from transformers import pipeline as hf_pipeline
    except ImportError as exc:
        raise ImportError(
            "HuggingFace model adapter requires torch and transformers. "
            "Install them with:  pip install torch transformers accelerate"
        ) from exc

    import torch as _torch

    _dtype = getattr(_torch, torch_dtype) if torch_dtype != "auto" else "auto"
    _pipe = hf_pipeline(
        "text-generation",
        model=model_name,
        device_map=device_map,
        torch_dtype=_dtype,
    )

    def _model_fn(prompt: str, n: int) -> list[str]:
        outputs = _pipe(
            prompt,
            num_return_sequences=n,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        return [out["generated_text"][len(prompt):].strip() for out in outputs]

    return _model_fn


# ---------------------------------------------------------------------------
# vLLM model factory (deferred to Stage 5)
# ---------------------------------------------------------------------------


def make_vllm_model(
    model_name: str,
    *,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 64,
    gpu_memory_utilization: float = 0.90,
) -> ModelFn:
    """Return a :data:`ModelFn` backed by a vLLM engine.

    vLLM batches all ``n`` sequences for a single prompt in one engine call,
    which is significantly faster than ``n`` sequential HF pipeline calls.
    Recommended for Stage 5 inference on A100 / H100 nodes.

    Requires ``vllm``.  Raises :class:`ImportError` with an install hint if
    the package is missing.

    Args:
        model_name: HuggingFace Hub model ID accepted by ``vllm.LLM``.
        temperature: Sampling temperature.
        top_p: Nucleus-sampling probability threshold.
        max_tokens: Maximum number of tokens to generate per answer.
        gpu_memory_utilization: Fraction of GPU memory to allocate for the KV
            cache (vLLM default is 0.90).

    Returns:
        A :data:`ModelFn` whose ``__call__`` signature is
        ``(prompt: str, n: int) -> list[str]``.

    Raises:
        ImportError: If ``vllm`` is not installed.
    """
    try:
        from vllm import LLM, SamplingParams  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "vLLM model adapter requires vllm. "
            "Install it with:  pip install vllm"
        ) from exc

    from vllm import LLM, SamplingParams

    _llm = LLM(
        model=model_name,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    _sampling_params = SamplingParams(
        n=1,  # set per-call below
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    def _model_fn(prompt: str, n: int) -> list[str]:
        params = _sampling_params.clone()
        params.n = n
        outputs = _llm.generate([prompt], params)
        return [out.text.strip() for out in outputs[0].outputs]

    return _model_fn


# ---------------------------------------------------------------------------
# NLI entailment function factory (deferred to Stage 5)
# ---------------------------------------------------------------------------


def make_nli_fn(
    model_name: str = "cross-encoder/nli-deberta-v3-base",
    *,
    device: str = "cuda",
    batch_size: int = 32,
) -> NliFn:
    """Return a :data:`NliFn` backed by a HuggingFace cross-encoder NLI model.

    The returned function calls the cross-encoder with the concatenated
    ``premise [SEP] hypothesis`` string and maps the model's output label to
    one of :data:`~semantic_entropy.clustering.NLI_ENTAILMENT`,
    :data:`~semantic_entropy.clustering.NLI_NEUTRAL`, or
    :data:`~semantic_entropy.clustering.NLI_CONTRADICTION`.

    Requires ``torch`` and ``transformers``.  Raises :class:`ImportError` with
    an install hint if either package is missing.

    Args:
        model_name: HuggingFace Hub model ID for the cross-encoder NLI model.
            The default (``cross-encoder/nli-deberta-v3-base``) is the model
            used in Farquhar et al. (2024).
        device: Device string passed to the HF pipeline (``"cuda"`` or
            ``"cpu"``).
        batch_size: Batch size for the HF pipeline.

    Returns:
        A :data:`NliFn` whose ``__call__`` signature is
        ``(premise: str, hypothesis: str) -> str``.

    Raises:
        ImportError: If ``torch`` or ``transformers`` are not installed.
    """
    try:
        from transformers import pipeline as hf_pipeline  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "NLI model adapter requires torch and transformers. "
            "Install them with:  pip install torch transformers"
        ) from exc

    from transformers import pipeline as hf_pipeline

    _pipe = hf_pipeline(
        "text-classification",
        model=model_name,
        device=device,
        batch_size=batch_size,
    )

    def _nli_fn(premise: str, hypothesis: str) -> str:
        result = _pipe(f"{premise} [SEP] {hypothesis}")[0]
        label = result["label"].lower()
        # Normalise to the three canonical constants
        if "entail" in label:
            return NLI_ENTAILMENT
        if "contradict" in label:
            from semantic_entropy.clustering import NLI_CONTRADICTION
            return NLI_CONTRADICTION
        return NLI_NEUTRAL

    return _nli_fn
