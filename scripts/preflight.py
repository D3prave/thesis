"""Pre-flight checks before submitting an HPC job.

Verifies, in order:

1. Python version matches ``pyproject.toml`` (>= 3.11).
2. Core package imports (``semantic_entropy``) without errors.
3. Optional dependency groups are importable for the features that
   will be exercised (``--features hf,kle,sep,vllm,plot,download``).
4. HuggingFace model weights are present in ``HF_HOME`` (for every
   ``--model`` value).
5. vLLM and TorchDynamo runtime settings are safe for NHR@FAU when
   ``--features`` includes ``vllm``.
6. Sentence-transformer embedding model is present (when
   ``--kle-model`` is given).
7. Data files referenced by ``--data-path`` exist and are readable.
8. The ``semantic-entropy-run-pipeline`` CLI accepts every feature
   flag we plan to use (``--kle-model``, ``--sep-probe``, etc.).

Each check prints either ``[ok]`` or ``[FAIL]`` and exits non-zero on
the first failure unless ``--no-strict`` is passed.

The script makes **no network calls** and runs in seconds; it is safe
to put in front of every sbatch.

Typical use (on the cluster, inside an interactive shell):

.. code-block:: bash

    module load python
    source ~/thesis/.venv/bin/activate

    python scripts/preflight.py \\
        --features hf,kle,sep \\
        --model mistralai/Mistral-7B-Instruct-v0.3 \\
        --kle-model sentence-transformers/all-MiniLM-L6-v2 \\
        --data-path data/processed/triviaqa_val_500.jsonl

Exit codes:
    0 - all checks passed
    1 - at least one check failed (only with default --strict)
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import sysconfig
from collections.abc import Iterable
from pathlib import Path

# ---------------------------------------------------------------------------
# Feature -> required modules / Python imports
# ---------------------------------------------------------------------------

FEATURE_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "download": ("huggingface_hub",),
    "hf": ("torch", "transformers"),
    "vllm": ("vllm",),
    "kle": ("numpy", "sentence_transformers"),
    "sep": ("numpy", "sklearn"),
    "plot": ("matplotlib",),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--features",
        default="hf",
        help=(
            "Comma-separated optional dependency groups to verify. "
            "One or more of: download,hf,vllm,kle,sep,plot,none. Default: hf."
        ),
    )
    p.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="HF_MODEL_ID",
        help=(
            "Verify that model weights are cached in HF_HOME. May be passed "
            "multiple times. An actual `from_pretrained` is NOT performed."
        ),
    )
    p.add_argument(
        "--kle-model",
        default=None,
        metavar="SENTENCE_TRANSFORMER_ID",
        help="If set, verify that the embedding model is cached.",
    )
    p.add_argument(
        "--data-path",
        action="append",
        default=[],
        metavar="JSONL",
        help=(
            "Data file to check for existence (may be passed multiple times)."
        ),
    )
    p.add_argument(
        "--no-strict",
        action="store_true",
        help=(
            "Do not exit on the first failure; continue and report a "
            "summary at the end."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Check primitives
# ---------------------------------------------------------------------------


class Reporter:
    """Tracks the number of failed checks for the final exit code."""

    def __init__(self, strict: bool) -> None:
        self.strict = strict
        self.failures = 0

    def ok(self, msg: str) -> None:
        print(f"  [ok]   {msg}", flush=True)

    def fail(self, msg: str) -> None:
        print(f"  [FAIL] {msg}", flush=True)
        self.failures += 1
        if self.strict:
            print(
                "\nAborting because --no-strict was not passed.",
                file=sys.stderr,
                flush=True,
            )
            sys.exit(1)


def _check_python_version(rep: Reporter) -> None:
    print("Python version", flush=True)
    major, minor = sys.version_info[:2]
    if (major, minor) >= (3, 11):
        rep.ok(f"Python {major}.{minor} >= 3.11")
    else:
        rep.fail(
            f"Python {major}.{minor} < 3.11 (pyproject.toml requires >= 3.11)"
        )


def _check_package_imports(rep: Reporter) -> None:
    print("Core package", flush=True)
    try:
        importlib.import_module("semantic_entropy")
        rep.ok("semantic_entropy importable")
    except ImportError as exc:
        rep.fail(f"semantic_entropy not importable: {exc}")


def _check_feature_imports(rep: Reporter, features: Iterable[str]) -> None:
    for feature in features:
        feature = feature.strip()
        if not feature or feature == "none":
            continue
        reqs = FEATURE_REQUIREMENTS.get(feature)
        if reqs is None:
            rep.fail(f"unknown feature group {feature!r}")
            continue
        print(f"Feature group [{feature}]", flush=True)
        for mod in reqs:
            try:
                importlib.import_module(mod)
                rep.ok(f"{mod} importable")
            except ImportError as exc:
                rep.fail(f"{mod} missing: {exc}")


def _check_hf_cache(rep: Reporter, model: str) -> None:
    print(f"HuggingFace cache for {model!r}", flush=True)
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    rep.ok(f"HF_HOME={hf_home}")
    cache_root = Path(hf_home) / "hub"
    # snapshot_download stores under models--<org>--<name>
    expected_dir = "models--" + model.replace("/", "--")
    candidate = cache_root / expected_dir
    if candidate.is_dir():
        rep.ok(f"weights present at {candidate}")
    else:
        rep.fail(
            f"weights NOT cached under {candidate}; "
            "run snapshot_download(repo_id=...) first"
        )


def _check_embedding_cache(rep: Reporter, model: str) -> None:
    print(f"Sentence-transformer cache for {model!r}", flush=True)
    # sentence-transformers re-uses HF_HOME under the hood.
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    cache_root = Path(hf_home) / "hub"
    expected_dir = "models--" + model.replace("/", "--")
    if (cache_root / expected_dir).is_dir():
        rep.ok(f"embedding model present at {cache_root / expected_dir}")
    else:
        rep.fail(
            f"embedding model NOT cached at {cache_root / expected_dir}; "
            "run SentenceTransformer(...) once with network access first"
        )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name}={raw!r} is not a recognized boolean")


def _python_h_available() -> tuple[bool, str]:
    include_dir = Path(sysconfig.get_path("include") or "")
    if (include_dir / "Python.h").is_file():
        return True, str(include_dir / "Python.h")
    for env_name in ("CPATH", "C_INCLUDE_PATH"):
        for raw_entry in os.environ.get(env_name, "").split(":"):
            if not raw_entry:
                continue
            candidate = Path(raw_entry) / "Python.h"
            if candidate.is_file():
                return True, str(candidate)
    return False, str(include_dir / "Python.h")


def _check_vllm_runtime_config(rep: Reporter, features: Iterable[str]) -> None:
    if "vllm" not in {feature.strip() for feature in features}:
        return

    print("vLLM runtime", flush=True)
    method = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
    if method == "spawn":
        rep.ok("VLLM_WORKER_MULTIPROC_METHOD=spawn")
    else:
        rep.fail(
            "VLLM_WORKER_MULTIPROC_METHOD must be spawn for CUDA-safe "
            f"multi-process startup; got {method!r}"
        )

    cuda_home = os.environ.get("CUDA_HOME")
    
    import shutil
    from pathlib import Path
    nvcc_path = shutil.which("nvcc")
    
    if not nvcc_path and cuda_home:
        candidate = Path(cuda_home) / "bin" / "nvcc"
        if candidate.is_file():
            nvcc_path = str(candidate)

    if nvcc_path:
        rep.ok(f"nvcc available at {nvcc_path}")
    else:
        rep.fail("nvcc NOT found on PATH or in CUDA_HOME/bin. flashinfer JIT compilation will fail. (Hint: run 'module load cuda/12.1.1' and 'export PATH=$CUDA_BASE/bin:$PATH')")

    if cuda_home:
        rep.ok(f"CUDA_HOME is set to {cuda_home}")
    else:
        rep.fail("CUDA_HOME is NOT set. flashinfer may fail to JIT compile.")

    try:
        enforce_eager = _env_bool("SE_VLLM_ENFORCE_EAGER", True)
    except ValueError as exc:
        rep.fail(str(exc))
        return

    available, detail = _python_h_available()

    if enforce_eager:
        rep.ok("SE_VLLM_ENFORCE_EAGER=1 disables vLLM's main compile path")
    elif available:
        rep.ok(f"Python.h available for vLLM compile path: {detail}")
    else:
        rep.fail(
            "SE_VLLM_ENFORCE_EAGER=0 but Python.h is not available to gcc "
            f"(checked {detail}); set SE_VLLM_ENFORCE_EAGER=1"
        )

    try:
        torch_compile_disabled = _env_bool("TORCH_COMPILE_DISABLE", False)
    except ValueError as exc:
        rep.fail(str(exc))
        return

    if torch_compile_disabled:
        rep.ok(
            "TORCH_COMPILE_DISABLE=1 disables residual torch.compile wrappers"
        )
    elif available:
        rep.ok(f"TORCH_COMPILE_DISABLE=0 but Python.h is available: {detail}")
    else:
        rep.fail(
            "TORCH_COMPILE_DISABLE must be 1 because vLLM can still call "
            "inner torch.compile wrappers even with enforce_eager enabled, "
            f"and Python.h is not available to gcc (checked {detail})"
        )


def _check_data_paths(rep: Reporter, paths: Iterable[str]) -> None:
    for p in paths:
        print(f"Data file {p!r}", flush=True)
        path = Path(p)
        if not path.is_file():
            rep.fail(f"file does not exist: {path}")
            continue
        try:
            with path.open("r", encoding="utf-8") as fh:
                first_line = fh.readline().strip()
        except OSError as exc:
            rep.fail(f"cannot open {path}: {exc}")
            continue
        if not first_line:
            rep.fail(f"{path} appears to be empty")
            continue
        rep.ok(f"{path} readable, first line {len(first_line)} bytes")


def _check_cli_flags(rep: Reporter) -> None:
    """Verify that the pipeline CLI exposes all extension flags.

    Importing the CLI module rather than shelling out keeps the check
    self-contained and avoids depending on console_scripts being on
    PATH.
    """
    print("Pipeline CLI flags", flush=True)
    try:
        from semantic_entropy.pipeline_cli import main as _main  # noqa: F401
    except ImportError as exc:
        rep.fail(f"semantic_entropy.pipeline_cli not importable: {exc}")
        return
    # Run --help in-process and check the help text. We capture stdout
    # via redirect_stdout to keep the preflight output clean.
    import contextlib
    import io

    from semantic_entropy import pipeline_cli

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                pipeline_cli.main(["--help"])
            except SystemExit:
                pass
    except Exception as exc:  # pragma: no cover - extremely unlikely
        rep.fail(f"pipeline_cli --help raised: {exc}")
        return

    help_text = buf.getvalue()
    expected_flags = [
        "--model-module",
        "--entailment-module",
        "--kle-model",
        "--kle-kernel",
        "--model-with-states-module",
        "--sep-probe",
    ]
    for flag in expected_flags:
        if flag in help_text:
            rep.ok(f"CLI accepts {flag}")
        else:
            rep.fail(f"CLI does NOT expose {flag}; reinstall the package?")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    rep = Reporter(strict=not args.no_strict)

    _check_python_version(rep)
    _check_package_imports(rep)

    features = [f for f in args.features.split(",") if f.strip()]
    _check_feature_imports(rep, features)
    _check_vllm_runtime_config(rep, features)

    for model in args.model:
        _check_hf_cache(rep, model)
    if args.kle_model:
        _check_embedding_cache(rep, args.kle_model)
    if args.data_path:
        _check_data_paths(rep, args.data_path)

    _check_cli_flags(rep)

    print("", flush=True)
    if rep.failures == 0:
        print("All preflight checks passed.")
        return 0
    print(f"{rep.failures} preflight check(s) failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
