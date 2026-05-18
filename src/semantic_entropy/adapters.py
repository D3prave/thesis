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
* ``SE_TEMPERATURE`` — sampling temperature (fallback: ``0.7``).
* ``SE_TOP_P`` — nucleus sampling threshold (fallback: ``0.95``).
* ``SE_MAX_NEW_TOKENS`` — max new tokens per sample (fallback: ``64``).
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
"""

from __future__ import annotations

import os

from semantic_entropy.models import (
    ModelFn,
    make_hf_model,
    make_nli_fn,
    make_vllm_model,
)

# Re-export the type for documentation
__all__ = [
    "make_local_test_model",
    "make_phase1_model",
    "make_vllm_phase1_model",
    "make_nli",
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
    (e.g. ``4`` for 70B on 4× A100, ``2`` for 70B on 2× H100-94GB).

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
        max_tokens=_env_int("SE_MAX_NEW_TOKENS", 64),
        gpu_memory_utilization=_env_float("SE_GPU_MEMORY_UTILIZATION", 0.90),
        tensor_parallel_size=_env_int("SE_TENSOR_PARALLEL_SIZE", 1),
        dtype=os.environ.get("SE_VLLM_DTYPE", "auto"),
        max_model_len=max_model_len,
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
