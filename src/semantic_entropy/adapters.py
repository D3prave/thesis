"""Concrete model adapter factories for the semantic-entropy pipeline.

Each public function in this module is a zero-argument factory that returns
a :data:`~semantic_entropy.models.ModelFn` or
:data:`~semantic_entropy.clustering.NliFn`.  They are designed to be passed
to the ``--model-module`` / ``--entailment-module`` flags of
``semantic-entropy-run-pipeline`` via the ``module:callable`` notation::

    # Local smoke test with a tiny 0.5B model (CPU / MPS, no GPU needed)
    semantic-entropy-run-pipeline \\
        --model-module semantic_entropy.adapters:make_local_test_model \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --dataset triviaqa --limit 2 \\
        /tmp/local_test_scored.jsonl

    # Phase 1 HPC: 7B/8B on a single GPU
    semantic-entropy-run-pipeline \\
        --model-module semantic_entropy.adapters:make_phase1_model \\
        --model mistralai/Mistral-7B-Instruct-v0.3 \\
        --dataset triviaqa \\
        --cluster alex \\
        results/phase1/tqa_scored.jsonl

    # Phase 2 HPC: add NLI clustering
    semantic-entropy-run-pipeline \\
        --model-module semantic_entropy.adapters:make_phase1_model \\
        --entailment-module semantic_entropy.adapters:make_nli \\
        --model mistralai/Mistral-7B-Instruct-v0.3 \\
        --dataset triviaqa \\
        --cluster alex \\
        results/phase2/tqa_nli_scored.jsonl

Environment variables
---------------------
All factories read configuration from environment variables so that Slurm
scripts can override them without editing Python code:

* ``SE_MODEL_NAME`` — HuggingFace model ID (fallback: factory default).
* ``SE_MODEL_REVISION`` — full model commit SHA (required for SEP collection).
* ``SE_TOKENIZER_REVISION`` — full tokenizer commit SHA (SEP; defaults to the
  model revision when both come from the same repository).
* ``SE_TEMPERATURE`` — sampling temperature (phase-1 fallback: ``0.7``;
  SEP fallback: paper-faithful ``1.0``).
* ``SE_TOP_P`` — nucleus sampling threshold (phase-1 fallback: ``0.95``;
  SEP fallback: paper-faithful ``0.9``).
* ``SE_TOP_K`` — sampling cap (SEP fallback: paper-faithful ``50``).
* ``SE_SEED`` — model sampling seed (fallback: ``0``).
* ``SE_MAX_NEW_TOKENS`` — max new tokens per sample (fallback: ``64``).
* ``SE_SEP_FEATURE_MAX_NEW_TOKENS`` — max new tokens for the separate greedy
  SEP feature response (fixed canonical fallback: ``256``).
* ``SE_SYSTEM_PROMPT`` — overrides the model system message. Unset keeps the
  terse QA default ("answer in a word or short phrase"); set it to a
  long-form instruction for the biography task (see
  :data:`BIO_SYSTEM_PROMPT`), or to ``"none"`` to disable the system message
  entirely. **Required for bio runs** — without it the model returns
  one-sentence answers and the long-form regime collapses.
* ``SE_DEVICE_MAP`` — device map string (fallback: ``"auto"``).
* ``SE_NLI_MODEL`` — NLI model ID (fallback: ``"cross-encoder/nli-deberta-v3-base"``).
* ``SE_NLI_DEVICE`` — NLI device (fallback: ``"cpu"``).
* ``SE_TENSOR_PARALLEL_SIZE`` — vLLM tensor-parallel degree for multi-GPU
  jobs (fallback: ``1``). Must equal the number of GPUs in the Slurm
  allocation.
* ``SE_GPU_MEMORY_UTILIZATION`` — vLLM ``gpu_memory_utilization`` (fallback:
  ``0.90``).
* ``SE_MAX_MODEL_LEN`` — optional cap on vLLM ``max_model_len`` (fallback:
  unset, defers to the model config).
* ``SE_VLLM_DTYPE`` — vLLM dtype string (fallback: ``"auto"``).
* ``SE_VLLM_ENFORCE_EAGER`` — when ``"1"`` (default) disables vLLM's
  ``torch.compile`` / CUDA-graph path. Required on NHR@FAU Alex
  compute nodes because they lack ``python3-devel`` and triton's runtime
  JIT compile of ``cuda_utils.c`` blows up with
  ``fatal error: Python.h: No such file or directory``. Pass
  ``SE_VLLM_ENFORCE_EAGER=0`` only on hosts where Python headers are
  reachable from gcc (e.g. a uv-managed Python or a python3-devel install).
* ``TORCH_COMPILE_DISABLE`` — should remain ``"1"`` on Alex, because
  vLLM can still trigger inner ``torch.compile`` wrappers even when
  ``enforce_eager`` is enabled.
"""

from __future__ import annotations

import os
import re

from semantic_entropy.models import (
    CANONICAL_SEP_FEATURE_MAX_NEW_TOKENS,
    ModelFn,
    ModelFnWithStates,
    make_hf_model,
    make_hf_model_with_states,
    make_nli_fn,
    make_vllm_model,
)

# Re-export the type for documentation
__all__ = [
    "make_local_test_model",
    "make_phase1_model",
    "make_phase1_model_with_states",
    "make_vllm_phase1_model",
    "make_nli",
    "BIO_SYSTEM_PROMPT",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    return float(raw) if raw is not None else default


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    return int(raw) if raw is not None else default


def _env_bool(key: str, default: bool) -> bool:
    """Parse a bool from an env var. Accepts 1/0, true/false, yes/no (case-insensitive)."""
    raw = os.environ.get(key)
    if raw is None:
        return default
    truthy = {"1", "true", "yes", "y", "on"}
    falsy = {"0", "false", "no", "n", "off"}
    lowered = raw.strip().lower()
    if lowered in truthy:
        return True
    if lowered in falsy:
        return False
    raise ValueError(
        f"Environment variable {key}={raw!r} is not a recognized boolean. "
        f"Use one of {sorted(truthy | falsy)}."
    )


def _sep_revisions() -> tuple[str, str]:
    """Read fail-closed, full Hugging Face revisions for SEP collection."""

    model_revision = os.environ.get("SE_MODEL_REVISION")
    tokenizer_revision = os.environ.get(
        "SE_TOKENIZER_REVISION", model_revision or ""
    )
    for key, value in (
        ("SE_MODEL_REVISION", model_revision),
        ("SE_TOKENIZER_REVISION", tokenizer_revision),
    ):
        if not isinstance(value, str) or not re.fullmatch(
            r"[0-9a-fA-F]{40}", value
        ):
            raise ValueError(
                f"{key} must be a full 40-character Hugging Face commit SHA "
                "for SEP collection"
            )
    return model_revision.lower(), tokenizer_revision.lower()


#: Recommended system prompt for the long-form biography task. The default
#: QA system prompt forces single-word/short-phrase answers, which collapses
#: the long-form regime: every "biography" comes back as one terse sentence,
#: so the M sampled paragraphs the long-form study depends on never exist.
#: Export ``SE_SYSTEM_PROMPT="$SE_BIO_SYSTEM_PROMPT"`` (or any custom string)
#: in the bio Slurm jobs so generation actually produces paragraph-length text.
BIO_SYSTEM_PROMPT = (
    "Write a concise, factual biography of the person named in the prompt. "
    "Cover their main life facts: when and where they were born, their "
    "nationality, their field or occupation, and their most notable "
    "achievements. Write a single coherent paragraph of about 120-150 words. "
    "State only facts you are confident are accurate."
)

#: Sentinel meaning "SE_SYSTEM_PROMPT was not set, keep the adapter default".
_SYSTEM_PROMPT_UNSET = object()


def _env_system_prompt() -> object | str | None:
    """Resolve an optional system-prompt override from ``SE_SYSTEM_PROMPT``.

    Returns :data:`_SYSTEM_PROMPT_UNSET` when the variable is absent (so the
    underlying ``make_*`` factory keeps its own default, the terse QA prompt).
    Returns ``None`` when set to an empty string or ``"none"`` (disable the
    system message entirely). Otherwise returns the literal string.
    """
    raw = os.environ.get("SE_SYSTEM_PROMPT")
    if raw is None:
        return _SYSTEM_PROMPT_UNSET
    if raw.strip().lower() in {"", "none"}:
        return None
    return raw


def _system_prompt_kwargs() -> dict[str, object]:
    """Build a ``{"system_prompt": ...}`` kwargs dict, or ``{}`` if unset."""
    resolved = _env_system_prompt()
    if resolved is _SYSTEM_PROMPT_UNSET:
        return {}
    return {"system_prompt": resolved}


# ---------------------------------------------------------------------------
# Local test adapter (tiny model, CPU/MPS, no GPU required)
# ---------------------------------------------------------------------------


def make_local_test_model() -> ModelFn:
    """Create a ModelFn from a tiny HF model for local smoke testing.

    Uses ``Qwen/Qwen2.5-0.5B-Instruct`` by default — small enough to run
    on a laptop CPU in a few seconds per prompt.  Override with
    ``SE_MODEL_NAME`` if desired.

    Returns:
        A :data:`ModelFn` backed by a HuggingFace text-generation pipeline.
    """
    model_name = os.environ.get("SE_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
    return make_hf_model(
        model_name,
        temperature=_env_float("SE_TEMPERATURE", 0.5),
        top_p=_env_float("SE_TOP_P", 0.95),
        max_new_tokens=_env_int("SE_MAX_NEW_TOKENS", 16),
        device_map="cpu",  # safe for any laptop
        torch_dtype="float32",  # CPU does not support bfloat16 on most Macs
    )


# ---------------------------------------------------------------------------
# Phase 1 HPC adapter (single GPU, 7B/8B)
# ---------------------------------------------------------------------------


def make_phase1_model() -> ModelFn:
    """Create a ModelFn for Phase 1 HPC jobs (single GPU, 7B/8B models).

    Reads ``SE_MODEL_NAME`` for the model ID.  Falls back to
    ``mistralai/Mistral-7B-Instruct-v0.3``.

    Returns:
        A :data:`ModelFn` backed by a HuggingFace text-generation pipeline
        with ``device_map="auto"`` and ``torch_dtype="auto"`` (bfloat16 on
        supported hardware).
    """
    model_name = os.environ.get(
        "SE_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3"
    )
    return make_hf_model(
        model_name,
        temperature=_env_float("SE_TEMPERATURE", 0.7),
        top_p=_env_float("SE_TOP_P", 0.95),
        max_new_tokens=_env_int("SE_MAX_NEW_TOKENS", 64),
        device_map=os.environ.get("SE_DEVICE_MAP", "auto"),
        torch_dtype="auto",
        **_system_prompt_kwargs(),
    )


# ---------------------------------------------------------------------------
# Phase 2 SEP adapter (HF with hidden-state extraction, single GPU)
# ---------------------------------------------------------------------------


def make_phase1_model_with_states() -> ModelFnWithStates:
    """Create a ModelFnWithStates for SEP hidden-state collection.

    Wraps :func:`semantic_entropy.models.make_hf_model_with_states` and
    forwards the same ``SE_*`` environment variables used by
    :func:`make_phase1_model`, plus two SEP-specific knobs:

    * ``SE_SEP_HIDDEN_LAYER`` — transformer layer to extract (default
      ``-1``: the last layer's output).
    * ``SE_SEP_HIDDEN_TOKEN`` — fixed to ``-1``, meaning the final response
      content token immediately before EOS.
    * ``SE_SEP_FEATURE_MAX_NEW_TOKENS`` — fixed to ``256`` for the separate
      greedy response; the stochastic sample cap remains
      ``SE_MAX_NEW_TOKENS``.

    Use this adapter via ``--model-with-states-module`` on the run-pipeline
    CLI when you need to collect hidden states for SEP probe training or
    inference. Single-GPU only — vLLM does not currently expose
    hidden-state hooks.
    """
    model_name = os.environ.get(
        "SE_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3"
    )
    model_revision, tokenizer_revision = _sep_revisions()
    return make_hf_model_with_states(
        model_name,
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        temperature=_env_float("SE_TEMPERATURE", 1.0),
        top_p=_env_float("SE_TOP_P", 0.9),
        top_k=_env_int("SE_TOP_K", 50),
        max_new_tokens=_env_int("SE_MAX_NEW_TOKENS", 64),
        feature_response_max_new_tokens=_env_int(
            "SE_SEP_FEATURE_MAX_NEW_TOKENS",
            CANONICAL_SEP_FEATURE_MAX_NEW_TOKENS,
        ),
        # None (the default) decodes the most-likely response greedily, which
        # is the sealed v2/v3 behaviour. Farquhar et al. instead *sample* it at
        # temperature 0.1, so replication runs set SE_FEATURE_TEMPERATURE=0.1.
        feature_temperature=(
            float(os.environ["SE_FEATURE_TEMPERATURE"])
            if os.environ.get("SE_FEATURE_TEMPERATURE", "").strip()
            else None
        ),
        device_map=os.environ.get("SE_DEVICE_MAP", "auto"),
        torch_dtype="auto",
        hidden_layer=_env_int("SE_SEP_HIDDEN_LAYER", -1),
        hidden_token=_env_int("SE_SEP_HIDDEN_TOKEN", -1),
        seed=_env_int("SE_SEED", 0),
        # v3 additions. Defaults preserve the sealed v2 record shape exactly:
        # logprob capture must be requested explicitly (SE_RETURN_LOGPROBS=1)
        # and feature capture stays on unless disabled for non-canonical
        # temperature collections (SE_SEP_COLLECT_FEATURES=0).
        return_logprobs=_env_bool("SE_RETURN_LOGPROBS", False),
        collect_features=_env_bool("SE_SEP_COLLECT_FEATURES", True),
        **_system_prompt_kwargs(),
    )


# ---------------------------------------------------------------------------
# Phase 1 HPC adapter (vLLM, single- or multi-GPU)
# ---------------------------------------------------------------------------


def make_vllm_phase1_model() -> ModelFn:
    """Create a vLLM-backed ModelFn for Phase 1 HPC jobs.

    Reads ``SE_MODEL_NAME`` for the model ID. Falls back to
    ``mistralai/Mistral-7B-Instruct-v0.3``. Reads ``SE_TENSOR_PARALLEL_SIZE``
    for the number of GPUs (default ``1``); set this to the GPU count of
    your Slurm allocation when running larger models with tensor parallelism
    (e.g. ``4`` for 70B on 4× A100).

    Returns:
        A :data:`ModelFn` backed by a ``vllm.LLM`` engine.
    """
    model_name = os.environ.get(
        "SE_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3"
    )

    max_model_len_raw = os.environ.get("SE_MAX_MODEL_LEN")
    max_model_len = int(max_model_len_raw) if max_model_len_raw else None

    return make_vllm_model(
        model_name,
        temperature=_env_float("SE_TEMPERATURE", 0.7),
        top_p=_env_float("SE_TOP_P", 0.95),
        # -1 disables top-k. The clean rerun sets SE_TOP_K=50 to match the
        # paper's nucleus(p=0.9)+top_k=50 sampling.
        top_k=_env_int("SE_TOP_K", -1),
        max_tokens=_env_int("SE_MAX_NEW_TOKENS", 64),
        gpu_memory_utilization=_env_float("SE_GPU_MEMORY_UTILIZATION", 0.90),
        tensor_parallel_size=_env_int("SE_TENSOR_PARALLEL_SIZE", 1),
        dtype=os.environ.get("SE_VLLM_DTYPE", "auto"),
        max_model_len=max_model_len,
        # Default ON because NHR@FAU compute nodes have no python3-devel and
        # triton's JIT compile of cuda_utils.c fails with Python.h missing.
        # See logs/phase1_70b_alex_3624187 for the canonical failure mode.
        enforce_eager=_env_bool("SE_VLLM_ENFORCE_EAGER", True),
        # Capture per-sequence logprobs by default so source generations carry
        # sequence_logprobs for naive_entropy / semantic_entropy_full. Cheap
        # (logprobs=1) and required for a faithful Fig. 2.
        return_logprobs=_env_bool("SE_RETURN_LOGPROBS", True),
        # Forward the Slurm seed so different seed submissions produce
        # genuinely different samples. SE_SEED is set by the sbatch scripts
        # from the $SEED variable (e.g. SEED=42/43/44).
        seed=_env_int("SE_SEED", 0),
        **_system_prompt_kwargs(),
    )


# ---------------------------------------------------------------------------
# NLI entailment adapter
# ---------------------------------------------------------------------------


def make_nli():
    """Create an NliFn for bidirectional entailment clustering.

    Uses ``cross-encoder/nli-deberta-v3-base`` by default.  Override with
    ``SE_NLI_MODEL`` and ``SE_NLI_DEVICE``.

    Returns:
        An :data:`NliFn` backed by a HuggingFace cross-encoder pipeline.
    """
    nli_model = os.environ.get(
        "SE_NLI_MODEL", "cross-encoder/nli-deberta-v3-base"
    )
    nli_device = os.environ.get("SE_NLI_DEVICE", "cpu")
    return make_nli_fn(model_name=nli_model, device=nli_device)
