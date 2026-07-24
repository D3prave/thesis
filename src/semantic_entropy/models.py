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

import copy
import hashlib
import math
import random
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

# Re-export the canonical type alias so callers can import from one place.
from semantic_entropy.clustering import NLI_ENTAILMENT, NLI_NEUTRAL, NliFn

CANONICAL_SEP_FEATURE_MAX_NEW_TOKENS = 256

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


@dataclass(frozen=True)
class SEPGeneration:
    """Output of a paper-faithful Semantic Entropy Probe collection call.

    ``sampled_answers`` are the high-temperature responses used only to
    calculate semantic entropy. ``greedy_answer`` and ``hidden_state`` come
    from a separate greedy response and supply the single SEP feature.
    Keeping these objects distinct prevents a stochastic DSE sample (or an
    average over samples) from being mislabeled as the SEP feature.
    """

    sampled_answers: list[str]
    greedy_answer: str
    hidden_state: list[float] | None
    metadata: dict[str, Any]
    #: Optional per-sample length-normalized log-probabilities (arithmetic
    #: mean raw-model token log-prob over the generated tokens up to and
    #: including the terminating EOS/EOT). Populated by the v3 collection
    #: protocol (``SE_RETURN_LOGPROBS=1``); ``None`` reproduces the sealed
    #: v2 record shape exactly.
    sequence_logprobs: list[float] | None = None


def _stable_prompt_seed(seed: int, prompt: str) -> int:
    """Derive a stable sampling seed independent of Python hash randomization."""
    return int.from_bytes(
        hashlib.sha256(f"{seed}\0{prompt}".encode("utf-8")).digest()[:8],
        byteorder="big",
    ) % (2**63 - 1)


def _final_content_token_offset(
    generated_token_ids: Sequence[int], special_token_ids: set[int]
) -> int:
    """Return the final non-special generated-token offset before EOS/padding."""
    content_length = len(generated_token_ids)
    while (
        content_length > 0
        and int(generated_token_ids[content_length - 1]) in special_token_ids
    ):
        content_length -= 1
    if content_length == 0:
        raise RuntimeError("greedy SEP response contains no content token before EOS")
    return content_length - 1


def _require_full_hf_revision(value: str | None, *, field: str) -> str:
    """Return a full Hugging Face commit SHA or fail before model loading."""

    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", value):
        raise ValueError(
            f"{field} must be a full 40-character Hugging Face commit SHA"
        )
    return value.lower()


def _require_eos_or_eot_termination(
    generated_token_ids: Sequence[int],
    *,
    termination_token_ids: set[int],
    pad_token_ids: set[int],
) -> int:
    """Return the terminating EOS/EOT ID or reject a max-token truncation."""

    if not generated_token_ids:
        raise RuntimeError("greedy SEP response generated no tokens")
    index = len(generated_token_ids) - 1
    while (
        index >= 0
        and int(generated_token_ids[index]) in pad_token_ids
        and int(generated_token_ids[index]) not in termination_token_ids
    ):
        index -= 1
    if index < 0 or int(generated_token_ids[index]) not in termination_token_ids:
        raise RuntimeError(
            "greedy SEP response did not terminate with EOS/EOT; refusing to "
            "label a max_new_tokens truncation as a final-content-token feature"
        )
    return int(generated_token_ids[index])


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
    model_revision: str,
    tokenizer_revision: str,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
    max_new_tokens: int = 64,
    feature_response_max_new_tokens: int = CANONICAL_SEP_FEATURE_MAX_NEW_TOKENS,
    device_map: str = "auto",
    torch_dtype: str = "auto",
    hidden_layer: int = -1,
    hidden_token: int = -1,
    seed: int = 0,
    system_prompt: str | None = (
        "Answer the following question as briefly as possible. "
        "Give only the answer — a word or short phrase — with no explanation."
    ),
    return_logprobs: bool = False,
    collect_features: bool = True,
) -> Callable[[str, int], SEPGeneration]:
    """Return a Kossen-style SEP collection function backed by HuggingFace.

    Each call performs two deliberately separate operations:

    * draw ``n`` responses with the semantic-entropy sampling configuration
      (Kossen et al.: temperature 1.0, top-p 0.9, top-k 50); and
    * generate one greedy response, then run a full forward pass over that
      response to extract the final content token's hidden state.

    The separate greedy response is the only probe feature. The stochastic
    responses are returned only for semantic-entropy calculation.

    Hidden-state semantics:

    * ``hidden_layer = -1`` selects the *final* transformer block's
      output (the layer immediately before the LM head).
    * ``hidden_token`` is fixed to ``-1`` and means the final non-special
      response token immediately before EOS/EOT. Other values are rejected so a
      run cannot silently change the preregistered feature position.

    The factory generates the ``n`` independent samples in a single
    ``generate`` call with ``num_return_sequences=n`` so the per-prompt
    forward-pass cost is shared.

    Args:
        model_name: HuggingFace Hub model ID.
        model_revision: Full Hugging Face commit SHA for the model weights.
        tokenizer_revision: Full Hugging Face commit SHA for the tokenizer.
        temperature: Sampling temperature.
        top_p: Nucleus-sampling probability threshold.
        max_new_tokens: Maximum number of tokens for each stochastic response.
        feature_response_max_new_tokens: Maximum number of tokens for the
            separate greedy feature response. The response must still end in
            EOS/EOT; reaching this ceiling fails closed.
        device_map: Passed to ``AutoModelForCausalLM.from_pretrained``.
        torch_dtype: Torch dtype string (``"auto"`` selects bfloat16 on
            supported hardware).
        hidden_layer: Transformer block index to extract; negative
            indices count from the end (``-1`` = last block).
        hidden_token: Must be ``-1`` (final content token before EOS/EOT).
        seed: Base sampling seed. A stable prompt-specific seed is derived
            from this value so results do not depend on prompt iteration order.
        system_prompt: Optional system message; same semantics as in
            :func:`make_hf_model`.
        return_logprobs: When true, additionally return per-sample
            length-normalized sequence log-probabilities computed from the
            RAW (untempered) model distribution — ``log_softmax`` of the
            generation logits, gathered at the sampled tokens, arithmetic
            mean over the generated tokens up to and including the
            terminating EOS/EOT. This is the jlko/semantic_uncertainty
            convention and is deliberately temperature-independent so that
            values are comparable across the v3 temperature family. Default
            false, which reproduces the sealed v2 record shape exactly.
        collect_features: When false, skip SEP feature extraction entirely:
            the greedy response is still generated (it supplies the
            most-likely answer for the paper accuracy rule and the P(True)
            prompt) but no forward pass is run and no hidden state or SEP
            feature metadata is produced. Used by v3 non-canonical
            temperature collections (the probe feature is greedy and hence
            temperature-independent; it is captured only at the canonical
            temperature).

    Returns:
        A callable returning :class:`SEPGeneration`.

    Raises:
        ImportError: If ``torch`` or ``transformers`` are not installed.
    """
    _model_revision = _require_full_hf_revision(
        model_revision, field="model_revision"
    )
    _tokenizer_revision = _require_full_hf_revision(
        tokenizer_revision, field="tokenizer_revision"
    )
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens <= 0
    ):
        raise ValueError("max_new_tokens must be a positive integer")
    if (
        isinstance(feature_response_max_new_tokens, bool)
        or not isinstance(feature_response_max_new_tokens, int)
        or feature_response_max_new_tokens <= 0
    ):
        raise ValueError("feature_response_max_new_tokens must be a positive integer")
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "HuggingFace model adapter with hidden states requires torch "
            "and transformers. Install them with: "
            "pip install torch transformers accelerate"
        ) from exc

    import torch as _torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _dtype = getattr(_torch, torch_dtype) if torch_dtype != "auto" else "auto"
    _tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=_tokenizer_revision
    )

    _model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=_model_revision,
        device_map=device_map,
        dtype=_dtype,
    )
    _model.eval()

    if _tokenizer.pad_token_id is None:
        fallback_eos = _tokenizer.eos_token_id
        if fallback_eos is None:
            model_eos = _model.generation_config.eos_token_id
            fallback_eos = (
                model_eos[0]
                if isinstance(model_eos, (list, tuple)) and model_eos
                else model_eos
            )
        if fallback_eos is None:
            raise ValueError(
                "canonical SEP collection requires a tokenizer pad token or "
                "a pinned model/tokenizer EOS/EOT token"
            )
        _tokenizer.pad_token_id = int(fallback_eos)

    if hidden_token != -1:
        raise ValueError(
            "Kossen-style SEP fixes hidden_token=-1, meaning the final "
            "content token immediately before EOS/EOT"
        )

    # vLLM convention: top_k <= 0 disables the cap; HuggingFace uses top_k=0 to
    # disable. Translate so a value of 50 matches the vLLM grid exactly and -1
    # means "no cap" (rather than silently falling back to HF's default of 50).
    _hf_top_k = top_k if top_k and top_k > 0 else 0
    # Preserve model-specific EOS/EOT and chat-generation behavior. Building a
    # fresh GenerationConfig would silently discard model-repository defaults.
    _sample_config = copy.deepcopy(_model.generation_config)
    _sample_config.do_sample = True
    _sample_config.temperature = temperature
    _sample_config.top_p = top_p
    _sample_config.top_k = _hf_top_k
    _sample_config.max_new_tokens = max_new_tokens
    _sample_config.pad_token_id = _tokenizer.pad_token_id
    if return_logprobs:
        # Raw-logit capture for sequence_logprobs. output_logits (not
        # output_scores) returns the UNPROCESSED logits, before the
        # temperature/top-p/top-k warpers — the paper-canonical re-scoring
        # distribution, comparable across sampling temperatures.
        _sample_config.return_dict_in_generate = True
        _sample_config.output_logits = True

    _greedy_config = copy.deepcopy(_model.generation_config)
    _greedy_config.do_sample = False
    _greedy_config.max_new_tokens = feature_response_max_new_tokens
    _greedy_config.pad_token_id = _tokenizer.pad_token_id
    _greedy_config.return_dict_in_generate = True

    def _token_id_set(value: Any) -> set[int]:
        if value is None:
            return set()
        if isinstance(value, int):
            return {value}
        return {int(item) for item in value}

    _termination_ids = _token_id_set(_greedy_config.eos_token_id)
    _termination_ids.update(_token_id_set(_tokenizer.eos_token_id))
    if not _termination_ids:
        raise ValueError(
            "canonical SEP collection requires an EOS/EOT token in the pinned "
            "model or tokenizer generation configuration"
        )
    _pad_ids = _token_id_set(_greedy_config.pad_token_id)

    def _model_fn(prompt: str, n: int) -> SEPGeneration:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        formatted = _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = _tokenizer(
            formatted,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(_model.device)
        prompt_len = inputs["input_ids"].shape[1]

        prompt_seed = _stable_prompt_seed(seed, prompt)
        _torch.manual_seed(prompt_seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(prompt_seed)

        with _torch.no_grad():
            sampled_out = _model.generate(
                **inputs,
                num_return_sequences=n,
                generation_config=_sample_config,
            )
            greedy_out = _model.generate(
                **inputs,
                num_return_sequences=1,
                generation_config=_greedy_config,
            )

        sampled_sequences = getattr(sampled_out, "sequences", sampled_out)
        sampled_ids = sampled_sequences[:, prompt_len:]
        sampled_answers = _tokenizer.batch_decode(
            sampled_ids, skip_special_tokens=True,
        )

        sequence_logprobs: list[float] | None = None
        if return_logprobs:
            # Per-step raw log-probabilities of the sampled tokens. Processed
            # step by step to avoid materializing (steps, n, vocab) at once.
            step_logprob_rows = []
            for step_index, step_logits in enumerate(sampled_out.logits):
                step_token_ids = sampled_ids[:, step_index]
                log_probs = _torch.log_softmax(
                    step_logits.to(_torch.float32), dim=-1
                )
                step_logprob_rows.append(
                    log_probs.gather(
                        1, step_token_ids.unsqueeze(1).to(log_probs.device)
                    ).squeeze(1).cpu()
                )
            token_logprobs = _torch.stack(step_logprob_rows, dim=1).tolist()
            sequence_logprobs = []
            for row_ids, row_logprobs in zip(sampled_ids.tolist(), token_logprobs):
                # Content length: tokens up to and including the first
                # EOS/EOT; padding after termination is excluded. A row with
                # no terminator ran to the cap and counts in full.
                length = len(row_ids)
                for index, token_id in enumerate(row_ids):
                    if int(token_id) in _termination_ids:
                        length = index + 1
                        break
                if length == 0:
                    raise RuntimeError(
                        "sampled sequence contains no generated tokens"
                    )
                value = math.fsum(row_logprobs[:length]) / length
                if not math.isfinite(value):
                    raise RuntimeError(
                        "non-finite sequence log-probability; refusing to "
                        "write an invalid record (fail-closed)"
                    )
                sequence_logprobs.append(float(value))
        sampled_eos_or_eot_count = 0
        for token_row in sampled_ids.tolist():
            try:
                _require_eos_or_eot_termination(
                    token_row,
                    termination_token_ids=_termination_ids,
                    pad_token_ids=_pad_ids,
                )
            except RuntimeError:
                continue
            sampled_eos_or_eot_count += 1

        greedy_sequence = greedy_out.sequences[0]
        generated_ids = greedy_sequence[prompt_len:]

        if not collect_features:
            # v3 non-canonical temperature collection: the greedy answer is
            # kept (most-likely answer for paper accuracy / P(True)) but no
            # SEP feature is extracted and no feature metadata is fabricated.
            # The record therefore carries no sep_* fields and remains valid
            # at any sampling temperature.
            greedy_answer = _tokenizer.decode(
                generated_ids.tolist(), skip_special_tokens=True
            )
            return SEPGeneration(
                sampled_answers=[answer.strip() for answer in sampled_answers],
                greedy_answer=greedy_answer.strip(),
                hidden_state=None,
                metadata={
                    "feature_capture": "disabled",
                    "sampling_temperature": float(temperature),
                    "sampling_top_p": float(top_p),
                    "sampling_top_k": int(top_k),
                    "sampling_num_responses": int(n),
                    "sampling_max_new_tokens": int(max_new_tokens),
                    "sampled_eos_or_eot_count": int(sampled_eos_or_eot_count),
                    "sampled_max_new_tokens_count": int(
                        n - sampled_eos_or_eot_count
                    ),
                    "base_seed": int(seed),
                    "prompt_seed": int(prompt_seed),
                    "model_id": model_name,
                    "model_revision": _model_revision,
                    "tokenizer_id": model_name,
                    "tokenizer_revision": _tokenizer_revision,
                },
                sequence_logprobs=sequence_logprobs,
            )

        greedy_degenerate = False
        terminating_token_id = None
        final_content_offset = None
        feature_position = None
        feature = None
        try:
            terminating_token_id = _require_eos_or_eot_termination(
                generated_ids.tolist(),
                termination_token_ids=_termination_ids,
                pad_token_ids=_pad_ids,
            )
        except RuntimeError:
            # Greedy decoding never emitted EOS/EOT within the feature-response
            # budget (degeneration / repetition looping). The fail-closed guard
            # correctly refuses to fabricate an SLT feature from a truncation;
            # rather than abort the whole shard, mark this one prompt's probe
            # feature as excluded and keep the record (and its stochastic
            # samples) so the entropy analysis is unaffected.
            greedy_degenerate = True

        if greedy_degenerate:
            greedy_answer = _tokenizer.decode(
                generated_ids.tolist(), skip_special_tokens=True
            )
        else:
            special_ids = set(_tokenizer.all_special_ids) | _termination_ids | _pad_ids
            final_content_offset = _final_content_token_offset(
                generated_ids.tolist(), special_ids
            )
            content_length = final_content_offset + 1

            # A generation-step hidden state predicts the next token and is easy
            # to shift by one. Re-run the complete greedy sequence through the
            # model and index the actual final content token explicitly.
            feature_position = prompt_len + content_length - 1
            feature_input = greedy_sequence[: feature_position + 1].unsqueeze(0)
            with _torch.no_grad():
                forward_out = _model(
                    input_ids=feature_input,
                    attention_mask=_torch.ones_like(feature_input),
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
            layer_count = len(forward_out.hidden_states)
            layer_idx = (
                hidden_layer if hidden_layer >= 0 else layer_count + hidden_layer
            )
            if layer_idx < 0 or layer_idx >= layer_count:
                raise ValueError(
                    f"hidden_layer {hidden_layer} is outside {layer_count} returned states"
                )
            feature = forward_out.hidden_states[layer_idx][0, feature_position, :]
            content_ids = generated_ids[:content_length]
            greedy_answer = _tokenizer.decode(content_ids, skip_special_tokens=True)

        return SEPGeneration(
            sampled_answers=[answer.strip() for answer in sampled_answers],
            greedy_answer=greedy_answer.strip(),
            hidden_state=(
                None
                if greedy_degenerate
                else feature.detach().to(_torch.float32).cpu().tolist()
            ),
            sequence_logprobs=sequence_logprobs,
            metadata={
                "protocol": "kossen_sep_2024",
                "feature_response_decoding": "greedy",
                "token_selection": "final_content_token_before_eos_or_eot",
                "layer_selection": "preregistered_final_layer",
                "generation_termination": "eos_or_eot",
                "termination_token_ids": sorted(_termination_ids),
                "terminating_special_token_id": (
                    None if greedy_degenerate else int(terminating_token_id)
                ),
                "generation_config_source": "model_generation_config_clone",
                "greedy_degenerate_excluded": bool(greedy_degenerate),
                "hidden_layer": hidden_layer,
                "hidden_dimension": (
                    None if greedy_degenerate else int(feature.shape[-1])
                ),
                "model_id": model_name,
                "model_revision": _model_revision,
                "tokenizer_id": model_name,
                "tokenizer_revision": _tokenizer_revision,
                "generated_content_token_index": (
                    None if greedy_degenerate else int(final_content_offset)
                ),
                "full_sequence_token_index": (
                    None if greedy_degenerate else int(feature_position)
                ),
                "sampling_temperature": float(temperature),
                "sampling_top_p": float(top_p),
                "sampling_top_k": int(top_k),
                "sampling_num_responses": int(n),
                "sampling_max_new_tokens": int(max_new_tokens),
                "feature_response_max_new_tokens": int(
                    feature_response_max_new_tokens
                ),
                "sampled_eos_or_eot_count": sampled_eos_or_eot_count,
                "sampled_max_new_tokens_count": int(n) - sampled_eos_or_eot_count,
                "base_seed": int(seed),
                "prompt_seed": int(prompt_seed),
            },
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
    seed: int | None = None,
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
        seed: Integer seed passed to ``vllm.LLM`` for reproducible sampling.
            When ``None`` (default), vLLM uses its own default (0), which means
            runs submitted without an explicit seed are not reproducible across
            jobs. Pass the same integer (e.g. 42, 43, 44) as the Slurm
            ``$SEED`` variable so different seed submissions genuinely produce
            different samples.
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
        # Pass seed so each Slurm seed submission produces genuinely
        # different samples and results are reproducible within a seed.
        "seed": seed if seed is not None else 0,
    }
    if max_model_len is not None:
        _llm_kwargs["max_model_len"] = max_model_len

    _llm = LLM(**_llm_kwargs)
    # Halt generation when an instruct model hallucinates a new chat turn.
    # Without these stops the model runs to max_tokens and emits template echo
    # or a fabricated next question; that contaminates the sampled answers, the
    # exact-match clustering, and the string-match correctness label (see
    # docs/DEEP_FINDINGS_REVERIFICATION_2026-06-15.md, generation-pollution
    # finding). These are literal strings the model emits (not special tokens),
    # so they are passed as string stops. "\n\n" is deliberately NOT included:
    # it would truncate paragraph-length biography answers.
    _STOP = [
        "[INST]", "[/INST]", "<s>", "</s>", "[CHAT]", "[/CHAT]",
        "<<SYS>>", "<</SYS>>", "[QUESTION]", "[ANSWER]",
    ]
    _sampling_params = SamplingParams(
        n=1,  # set per-call below
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,  # -1 disables; paper uses top_k=50 with nucleus p=0.9
        max_tokens=max_tokens,
        stop=_STOP,
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
        best_params = SamplingParams(
            n=1,
            temperature=0.1,
            top_p=1.0,
            max_tokens=max_tokens,
            stop=_STOP,
        )
        out = _llm.generate([formatted], best_params)
        return out[0].outputs[0].text.strip()

    def _batch_outputs(prompts: list[str], n: int) -> tuple[list[list[str]], list[list[str]]]:
        """Generate texts and termination reasons for one prompt batch."""

        tokenizer = _llm.get_tokenizer()
        formatted = [
            tokenizer.apply_chat_template(
                ([{"role": "system", "content": system_prompt}] if system_prompt else [])
                + [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
        params = _sampling_params.clone()
        params.n = n
        outputs = _llm.generate(formatted, params)
        answers = [[seq.text.strip() for seq in out.outputs] for out in outputs]
        finish_reasons = [
            [str(seq.finish_reason or "") for seq in out.outputs]
            for out in outputs
        ]
        return answers, finish_reasons

    def _batch(prompts: list[str], n: int) -> list[list[str]]:
        """Generate ``n`` samples for many prompts in a single vLLM call.

        vLLM batches the whole list with continuous batching, so this is
        dramatically faster than calling the per-prompt ``_model_fn`` in a loop
        (essential for large stages such as FactualBio answer regeneration with
        tens of thousands of prompts). Returns one answer list per input prompt,
        in input order. Texts only — logprobs are not returned by this path.
        """
        answers, _ = _batch_outputs(prompts, n)
        return answers

    def _batch_with_metadata(
        prompts: list[str], n: int
    ) -> tuple[list[list[str]], list[list[str]]]:
        """Return responses plus vLLM finish reasons without a second pass."""

        return _batch_outputs(prompts, n)

    _model_fn.best_answer = _best_answer  # type: ignore[attr-defined]
    _model_fn.batch = _batch  # type: ignore[attr-defined]
    _model_fn.batch_with_metadata = _batch_with_metadata  # type: ignore[attr-defined]
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
    revision: str | None = None,
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

    # revision pins the exact snapshot (v3 protocol requirement: the vault
    # cache can hold multiple snapshots of the same model id).
    _model = SentenceTransformer(model_name, device=device, revision=revision)

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
    revision: str | None = None,
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
        revision: Exact Hugging Face model revision when provenance is locked.
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

    pipeline_kwargs: dict[str, Any] = {
        "model": model_name,
        "device": device,
        "batch_size": batch_size,
    }
    if revision is not None:
        pipeline_kwargs["revision"] = revision
    _pipe = hf_pipeline("text-classification", **pipeline_kwargs)

    def _nli_fn(premise: str, hypothesis: str) -> str:
        raw = _pipe({"text": premise, "text_pair": hypothesis})
        # Transformers returns a single dict for a single input pair on current
        # versions, but a one-element list on others (or with top_k set).
        result = raw[0] if isinstance(raw, list) else raw
        label = result["label"].lower()
        # Normalize to the three canonical constants
        if "entail" in label:
            return NLI_ENTAILMENT
        if "contradict" in label:
            from semantic_entropy.clustering import NLI_CONTRADICTION
            return NLI_CONTRADICTION
        if "neutral" in label:
            return NLI_NEUTRAL
        raise ValueError(f"unexpected NLI label from pinned model: {result['label']!r}")

    return _nli_fn
