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
           "mistralai/Mistral-7B-Instruct-v0.3",
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

#: Type alias for a sampling model function that also returns hidden states.
#:
#: A ``ModelFnWithStates`` has signature ``(prompt, n) -> (answers, states)``
#: where ``answers`` is the same list of ``n`` strings produced by a
#: :data:`ModelFn`, and ``states`` is a ``(n, hidden_dim)`` matrix —
#: returned as ``list[list[float]]`` — of per-sample hidden states extracted
#: from the generation model.  The hidden state is typically the
#: last-token activation at a chosen transformer layer, and is used by
#: :mod:`semantic_entropy.probes` (Semantic Entropy Probes).
ModelFnWithStates = Callable[[str, int], tuple[list[str], list[list[float]]]]


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
    system_prompt: str | None = (
        "Answer the following question as briefly as possible. "
        "Give only the answer — a word or short phrase — with no explanation."
    ),
) -> ModelFn:
    """Return a :data:`ModelFn` backed by a HuggingFace text-generation pipeline.

    Requires ``torch`` and ``transformers``.  Raises :class:`ImportError` with
    an install hint if either package is missing.

    Args:
        model_name: HuggingFace Hub model ID, e.g.
            ``"mistralai/Mistral-7B-Instruct-v0.3"``.
        temperature: Sampling temperature.
        top_p: Nucleus-sampling probability threshold.
        max_new_tokens: Maximum number of tokens to generate per answer.
        device_map: Passed directly to the HF pipeline (``"auto"`` works for
            single-GPU and multi-GPU setups).
        torch_dtype: Torch dtype string (``"auto"`` selects bfloat16 on
            supported hardware, ``"float32"`` for CPU).
        system_prompt: Optional system message prepended to every prompt.
            The default instructs the model to give brief, phrase-only answers,
            which is required for the TriviaQA/SVAMP normalizers to match
            correctly.  Pass ``None`` to disable and use raw text completion
            (not recommended for instruct-tuned models).

    Returns:
        A :data:`ModelFn` whose ``__call__`` signature is
        ``(prompt: str, n: int) -> list[str]``.

    Raises:
        ImportError: If ``torch`` or ``transformers`` are not installed.
    """
    try:
        import torch  # noqa: F401
        from transformers import GenerationConfig
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
        dtype=_dtype,
    )

    # Build a GenerationConfig once so we never mix config-object +
    # keyword-arg styles (which triggers a deprecation warning in
    # transformers >= 4.40) and so that the model's own max_length default
    # (often 20) does not conflict with max_new_tokens.
    _gen_config = GenerationConfig(
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )

    def _model_fn(prompt: str, n: int) -> list[str]:
        # Build a messages list so the pipeline applies the model's chat
        # template (e.g. Mistral [INST]…[/INST], Llama-3 <|user|>…).
        # Without this, instruct-tuned models run in raw completion mode and
        # produce verbose sentences that fail the answer normalizers.
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        outputs = _pipe(
            messages,
            num_return_sequences=n,
            generation_config=_gen_config,
            return_full_text=False,
        )
        # When the pipeline receives a messages list it returns a list of
        # lists; each inner list contains one dict with key "generated_text"
        # that is itself a messages list — take the last message's content.
        results: list[str] = []
        for seq in outputs:
            text = seq["generated_text"]
            if isinstance(text, list):
                # Chat-template path: generated_text is a messages list
                text = text[-1]["content"]
            results.append(text.strip())
        return results

    return _model_fn


# ---------------------------------------------------------------------------
# HuggingFace model factory with hidden-state extraction (for SEP)
# ---------------------------------------------------------------------------


def make_hf_model_with_states(
    model_name: str,
    *,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_new_tokens: int = 64,
    device_map: str = "auto",
    torch_dtype: str = "auto",
    hidden_layer: int = -1,
    hidden_token: int = -1,
    system_prompt: str | None = (
        "Answer the following question as briefly as possible. "
        "Give only the answer — a word or short phrase — with no explanation."
    ),
) -> ModelFnWithStates:
    """Return a :data:`ModelFnWithStates` backed by HuggingFace.

    Wraps ``transformers.AutoModelForCausalLM.generate`` with
    ``output_hidden_states=True`` and returns, alongside each generated
    answer, the hidden state of the chosen token at the chosen
    transformer layer.  This is the data source used by Semantic Entropy
    Probes (see :mod:`semantic_entropy.probes`).

    Hidden-state semantics:

    * ``hidden_layer = -1`` selects the *final* transformer block's
      output (the layer immediately before the LM head).
    * ``hidden_token = -1`` selects the *last* generated token; ``-2``
      selects the second-to-last (the "second-to-last token" probe of
      Slobodkin et al., 2023).  Indices are negative-from-end.

    The factory generates the ``n`` independent samples in a single
    ``generate`` call with ``num_return_sequences=n`` so the per-prompt
    forward-pass cost is shared.

    Args:
        model_name: HuggingFace Hub model ID.
        temperature: Sampling temperature.
        top_p: Nucleus-sampling probability threshold.
        max_new_tokens: Maximum number of tokens to generate per answer.
        device_map: Passed to ``AutoModelForCausalLM.from_pretrained``.
        torch_dtype: Torch dtype string (``"auto"`` selects bfloat16 on
            supported hardware).
        hidden_layer: Transformer block index to extract; negative
            indices count from the end (``-1`` = last block).
        hidden_token: Position of the generated token whose hidden state
            is returned; negative indices count from the end of the
            generated continuation.
        system_prompt: Optional system message; same semantics as in
            :func:`make_hf_model`.

    Returns:
        A :data:`ModelFnWithStates`.

    Raises:
        ImportError: If ``torch`` or ``transformers`` are not installed.
    """
    try:
        import torch  # noqa: F401
        from transformers import (  # noqa: F401
            AutoModelForCausalLM,
            AutoTokenizer,
            GenerationConfig,
        )
    except ImportError as exc:
        raise ImportError(
            "HuggingFace model adapter with hidden states requires torch "
            "and transformers. Install them with: "
            "pip install torch transformers accelerate"
        ) from exc

    import torch as _torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        GenerationConfig,
    )

    _dtype = getattr(_torch, torch_dtype) if torch_dtype != "auto" else "auto"
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    if _tokenizer.pad_token_id is None:
        _tokenizer.pad_token_id = _tokenizer.eos_token_id

    _model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        dtype=_dtype,
    )
    _model.eval()

    _gen_config = GenerationConfig(
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_hidden_states=True,
        pad_token_id=_tokenizer.pad_token_id,
    )

    def _model_fn(prompt: str, n: int) -> tuple[list[str], list[list[float]]]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        formatted = _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = _tokenizer(formatted, return_tensors="pt").to(_model.device)
        prompt_len = inputs["input_ids"].shape[1]

        with _torch.no_grad():
            out = _model.generate(
                **inputs,
                num_return_sequences=n,
                generation_config=_gen_config,
            )

        # `out.sequences` shape: (n, prompt_len + new_len).
        # `out.hidden_states` is a tuple of length new_len; element t is a
        # tuple of (num_layers + 1) tensors, each shaped
        # (n, seq_len_at_step_t, hidden_dim). For the *first* step
        # seq_len_at_step_t = prompt_len; for subsequent steps it is 1.
        new_len = len(out.hidden_states)
        if new_len == 0:
            raise RuntimeError(
                "generate produced zero new tokens; cannot extract hidden states"
            )

        # Resolve hidden_token to a forward index over the generated continuation.
        token_idx = hidden_token if hidden_token >= 0 else new_len + hidden_token
        token_idx = max(0, min(token_idx, new_len - 1))
        layer_hidden_states = out.hidden_states[token_idx]
        layer_idx = hidden_layer if hidden_layer >= 0 else len(layer_hidden_states) + hidden_layer
        chosen_layer = layer_hidden_states[layer_idx]
        # For step > 0 the tensor is (n, 1, hidden_dim); for step 0 it's
        # (n, prompt_len, hidden_dim) — we always take the *last* position
        # within that step's tensor, which is the just-emitted token.
        per_sample_states = chosen_layer[:, -1, :]

        # Decode just the newly-generated tokens.
        generated_ids = out.sequences[:, prompt_len:]
        decoded = _tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True,
        )

        return (
            [d.strip() for d in decoded],
            per_sample_states.detach().to(_torch.float32).cpu().tolist(),
        )

    return _model_fn


# ---------------------------------------------------------------------------
# vLLM model factory (deferred to Stage 5)
# ---------------------------------------------------------------------------


def make_vllm_model(
    model_name: str,
    *,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = -1,
    max_tokens: int = 64,
    gpu_memory_utilization: float = 0.90,
    tensor_parallel_size: int = 1,
    dtype: str = "auto",
    max_model_len: int | None = None,
    enforce_eager: bool = True,
    return_logprobs: bool = False,
    system_prompt: str | None = (
        "Answer the following question as briefly as possible. "
        "Give only the answer — a word or short phrase — with no explanation."
    ),
) -> ModelFn:
    """Return a :data:`ModelFn` backed by a vLLM engine.

    vLLM batches all ``n`` sequences for a single prompt in one engine call,
    which is significantly faster than ``n`` sequential HF pipeline calls.
    Recommended for Stage 5 inference on Alex A100 nodes.

    Requires ``vllm``.  Raises :class:`ImportError` with an install hint if
    the package is missing.

    Args:
        model_name: HuggingFace Hub model ID accepted by ``vllm.LLM``.
        temperature: Sampling temperature.
        top_p: Nucleus-sampling probability threshold.
        max_tokens: Maximum number of tokens to generate per answer.
        gpu_memory_utilization: Fraction of GPU memory to allocate for the KV
            cache (vLLM default is 0.90).
        tensor_parallel_size: Number of GPUs over which to split the model
            via tensor parallelism. Defaults to ``1`` (single-GPU). For a
            70B model on Alex (4× A100-80GB) pass ``4``. Must equal the
            number of GPUs allocated to the Slurm job.
        dtype: Torch dtype string passed to ``vllm.LLM``. Defaults to
            ``"auto"`` (vLLM selects bfloat16 on supported hardware).
        max_model_len: Optional cap on the maximum context length.
            ``None`` defers to the model config. Setting an explicit cap
            (e.g. ``4096``) avoids vLLM raising on models whose declared
            ``max_position_embeddings`` exceeds the available KV-cache
            memory on the chosen GPUs.
        enforce_eager: When ``True`` (default), disable vLLM's main
            ``torch.compile`` / CUDA-graph code path and run model forwards in
            eager mode. On Alex this must be paired with
            ``TORCH_COMPILE_DISABLE=1`` from ``slurm/_common.sh`` because vLLM
            can still trigger inner ``torch.compile`` wrappers while profiling.
            The failure symptom is ``gcc: fatal error: Python.h: No such file
            or directory`` inside vLLM startup.
        system_prompt: Optional system message; same semantics as in
            :func:`make_hf_model`.

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

    _llm_kwargs: dict[str, object] = {
        "model": model_name,
        "gpu_memory_utilization": gpu_memory_utilization,
        "tensor_parallel_size": tensor_parallel_size,
        "dtype": dtype,
        "enforce_eager": enforce_eager,
    }
    if max_model_len is not None:
        _llm_kwargs["max_model_len"] = max_model_len

    _llm = LLM(**_llm_kwargs)
    _sampling_params = SamplingParams(
        n=1,  # set per-call below
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,  # -1 disables; paper uses top_k=50 with nucleus p=0.9
        max_tokens=max_tokens,
        # Requesting logprobs populates each output's cumulative_logprob, which
        # we length-normalize for the probability-weighted estimators.
        logprobs=1 if return_logprobs else None,
    )

    def _model_fn(prompt: str, n: int):
        # Apply chat template via vLLM's tokenizer so instruct models
        # receive the correct [INST]/system-prompt framing.
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        tokenizer = _llm.get_tokenizer()
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        params = _sampling_params.clone()
        params.n = n
        outputs = _llm.generate([formatted], params)
        completions = outputs[0].outputs
        answers = [out.text.strip() for out in completions]
        if not return_logprobs:
            return answers
        logprobs = [_length_normalized_logprob(out) for out in completions]
        return answers, logprobs

    def _best_answer(prompt: str) -> str:
        """Single low-temperature (T=0.1) "most likely" answer (paper accuracy).

        Reuses the same vLLM engine — no second model load. Used to assess
        accuracy on one point-estimate answer per the paper's protocol.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        formatted = _llm.get_tokenizer().apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        best_params = SamplingParams(n=1, temperature=0.1, top_p=1.0, max_tokens=max_tokens)
        out = _llm.generate([formatted], best_params)
        return out[0].outputs[0].text.strip()

    _model_fn.best_answer = _best_answer  # type: ignore[attr-defined]
    return _model_fn


def _length_normalized_logprob(completion: Any) -> float:
    """Mean per-token log-probability of a vLLM completion.

    ``cumulative_logprob`` is the summed token log-probability of the
    generated sequence; dividing by the number of generated tokens gives the
    length-normalized log-probability used by the paper's estimators. Empty
    generations fall back to the raw cumulative value (0.0 if unavailable).
    """
    cumulative = getattr(completion, "cumulative_logprob", None)
    if cumulative is None:
        return 0.0
    num_tokens = len(getattr(completion, "token_ids", []) or [])
    return float(cumulative) / num_tokens if num_tokens > 0 else float(cumulative)


# ---------------------------------------------------------------------------
# Sentence-embedding factory (deferred — used by Kernel Language Entropy)
# ---------------------------------------------------------------------------


def make_embedding_fn(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    *,
    device: str = "cpu",
    batch_size: int = 32,
    normalize_embeddings: bool = False,
):
    """Return a function mapping a list of strings to a 2D embedding matrix.

    The returned callable has signature
    ``(texts: Sequence[str]) -> list[list[float]]``.  Each row of the
    output is the dense embedding of the corresponding input string, as
    produced by ``sentence_transformers.SentenceTransformer.encode``.

    The factory is intended to feed
    :func:`semantic_entropy.kle.compute_kle`, which is content-free with
    respect to embedding model choice: any encoder that returns a
    fixed-dimensional vector per string will work, but the Phase 2
    protocol uses ``sentence-transformers/all-MiniLM-L6-v2`` as the
    standard backbone.

    Requires the ``sentence-transformers`` package.  Raises
    :class:`ImportError` with an install hint if it is not available.

    Args:
        model_name: HuggingFace model ID accepted by
            ``SentenceTransformer``.  Defaults to a small all-purpose
            CPU-friendly model.
        device: Device string passed to ``SentenceTransformer``.  KLE
            runs cheaply on CPU even for a few hundred prompts × 10
            samples, so the default is ``"cpu"`` to leave the GPU free
            for the generation model when both are co-located.
        batch_size: Encoding batch size.
        normalize_embeddings: If ``True`` the encoder returns unit-norm
            embeddings, which makes the RBF kernel equivalent to a
            cosine-similarity kernel up to a monotone transform.  Leave
            ``False`` for the default RBF behavior.

    Returns:
        A callable as described above.

    Raises:
        ImportError: If ``sentence_transformers`` is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Embedding adapter requires sentence-transformers. "
            "Install the optional kle dependencies with: "
            "pip install -e '.[kle]'"
        ) from exc

    _model = SentenceTransformer(model_name, device=device)

    def _embed_fn(texts: Sequence[str]):
        arr = _model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=False,
        )
        # Return Python lists to keep the type contract numpy-optional.
        return arr.tolist()

    return _embed_fn


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

    The returned function calls the cross-encoder with separate premise and
    hypothesis fields and maps the model's output label to one of
    :data:`~semantic_entropy.clustering.NLI_ENTAILMENT`,
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
        result = _pipe({"text": premise, "text_pair": hypothesis})[0]
        label = result["label"].lower()
        # Normalize to the three canonical constants
        if "entail" in label:
            return NLI_ENTAILMENT
        if "contradict" in label:
            from semantic_entropy.clustering import NLI_CONTRADICTION
            return NLI_CONTRADICTION
        return NLI_NEUTRAL

    return _nli_fn
