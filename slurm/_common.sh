# ============================================================================
# slurm/_common.sh — shared environment preamble for every GPU job on Alex.
#
# Source this file *after* `set -euo pipefail` and *before* any `python` call:
#
#     REPO_ROOT="${SLURM_SUBMIT_DIR}"
#     # shellcheck disable=SC1091
#     source "${REPO_ROOT}/slurm/_common.sh"
#
# It is intentionally NOT an sbatch script. The job script owns #SBATCH
# directives; this file owns the *runtime* environment.
#
# Why a shared preamble exists
# ----------------------------
# Phase 1 70B job 3624187 burned ~6 minutes of a 4×A100 reservation before
# vLLM crashed with `Python.h: No such file or directory`. Root cause: NHR@FAU
# Alex compute nodes run AlmaLinux 8 without python3-devel, so triton's
# runtime JIT compile of cuda_utils.c (driven by torch.compile / inductor
# combo-kernel benchmarking) fails. This preamble configures every knob
# needed to either (a) avoid that code path entirely or (b) make the JIT
# compile actually succeed when Python headers happen to be reachable.
#
# What it sets, in one place, for ALL jobs:
#   * HuggingFace cache → vault (HF_HOME, HF_HUB_CACHE, TRANSFORMERS_CACHE).
#   * Triton + Inductor caches → $WORK (persistent across jobs) for warm
#     starts, with scratch on $TMPDIR (node-local NVMe).
#   * CPATH best-effort pointing at the venv's Python include dir so triton
#     can find Python.h when one exists (uv-managed Python ships it).
#   * OMP_NUM_THREADS / SRUN_CPUS_PER_TASK aligned with SLURM_CPUS_PER_TASK
#     (Slurm >22.05 doesn't propagate the latter automatically).
#   * VLLM_WORKER_MULTIPROC_METHOD=spawn (required for tensor-parallel
#     fork-safety with CUDA).
#   * SE_VLLM_ENFORCE_EAGER=1 by default — opt out only on hosts proven to
#     have Python.h (e.g. an interactive node where you've verified it).
#   * HF_HUB_OFFLINE=1 so jobs fail fast on a missing model cache rather
#     than discovering it inside a 6-hour A100 reservation.
#
# The preamble assumes:
#   * REPO_ROOT is already set (e.g. to $SLURM_SUBMIT_DIR).
#   * A .venv exists at $REPO_ROOT/.venv with the project installed.
# ============================================================================

# --- 0. Sanity ---------------------------------------------------------------
if [ -z "${REPO_ROOT:-}" ]; then
    echo "ERROR: _common.sh sourced without REPO_ROOT set." >&2
    return 1 2>/dev/null || exit 1
fi
if [ ! -d "${REPO_ROOT}/.venv" ]; then
    echo "ERROR: ${REPO_ROOT}/.venv not found. Create it on a login node with:" >&2
    echo "       module load python && python -m venv .venv" >&2
    echo "       source .venv/bin/activate && pip install -e '.[vllm,hf]'" >&2
    return 1 2>/dev/null || exit 1
fi

# --- 1. Modules + venv -------------------------------------------------------
module load python 2>/dev/null || true   # may already be loaded
# shellcheck disable=SC1091
source "${REPO_ROOT}/.venv/bin/activate"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# --- 2. HuggingFace cache ----------------------------------------------------
# /home/vault is backed up and survives the account lifetime (500 GB quota,
# we use ~670 GB already → request a quota extension or move 70B to $WORK).
export HF_HOME="${HF_HOME:-/home/vault/b192aa/b192aa36/huggingface}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
export SENTENCE_TRANSFORMERS_HOME="${HF_HOME}/sentence-transformers"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${SENTENCE_TRANSFORMERS_HOME}"
# Fail fast if a job asks for a model that isn't on disk. Without this,
# the compute node tries a download → blocked (no proxy by default) →
# wastes the entire allocation.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# --- 3. Triton + Inductor caches --------------------------------------------
# $WORK has no backup but no inode-tight quota and survives across jobs, so
# a warm triton cache cuts vLLM startup by ~30s on subsequent runs.
# If $WORK isn't set (rare), fall back to vault.
_SE_WORK="${WORK:-/home/vault/b192aa/b192aa36}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${_SE_WORK}/.cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${_SE_WORK}/.cache/torchinductor}"
# vLLM's own compile cache also goes here (default is ~/.cache/vllm, which
# eats the tiny 50 GB $HOME quota fast).
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${_SE_WORK}/.cache/vllm}"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}"
# Use the node-local SSD for build-time scratch (Alex: 14 TB NVMe).
# $TMPDIR is set by SLURM per-job and wiped at job end.
if [ -n "${TMPDIR:-}" ]; then
    export TMPDIR
    export TRITON_DUMP_DIR="${TMPDIR}/triton-dump"
    mkdir -p "${TRITON_DUMP_DIR}"
fi

# --- 4. Best-effort Python.h discovery --------------------------------------
# Triton's CudaUtils() compiles cuda_utils.c at runtime and the resulting
# command line embeds whichever path `python3-config --includes` reports.
# On Alex's RHEL-8 system Python the resulting path
# `/usr/include/python3.11/Python.h` does NOT exist (no python3-devel).
# If the venv (or another Python on PATH) ships its own headers we can
# point gcc at them via CPATH so the JIT compile succeeds.
_SE_PY_INCLUDE="$(python -c 'import sysconfig; print(sysconfig.get_path("include"))' 2>/dev/null || true)"
if [ -n "${_SE_PY_INCLUDE}" ] && [ -f "${_SE_PY_INCLUDE}/Python.h" ]; then
    export CPATH="${_SE_PY_INCLUDE}${CPATH:+:${CPATH}}"
    export C_INCLUDE_PATH="${_SE_PY_INCLUDE}${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}"
    echo "[_common.sh] Python.h found at ${_SE_PY_INCLUDE}; added to CPATH."
else
    echo "[_common.sh] Python.h NOT findable from $(which python). Relying on enforce_eager."
fi

# --- 5. CPU threading --------------------------------------------------------
# Slurm >22.05 doesn't propagate --cpus-per-task to srun; set both.
# vLLM also warns when OMP_NUM_THREADS is unset and reduces torch parallelism
# to 1 thread, costing tokens/s on the prefill path.
if [ -n "${SLURM_CPUS_PER_TASK:-}" ]; then
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK}}"
    export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"
fi
# AMD Zen3 (Alex) likes mkl-on-amd workaround disabled (NHR@FAU explicitly
# does not recommend the older MKL_DEBUG_CPU_TYPE=5 hack any more), but a
# correct value of MKL_CBWR=AUTO is still helpful.
export MKL_CBWR="${MKL_CBWR:-AUTO}"

# --- 6. vLLM knobs -----------------------------------------------------------
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
# Default ON: skip torch.compile / CUDA graphs so we never enter triton's
# JIT compile path. ~10–30% slower than compiled, but bullet-proof on Alex.
# Override with --export=...,SE_VLLM_ENFORCE_EAGER=0 only after preflight
# proves the JIT path actually works on the node you got.
export SE_VLLM_ENFORCE_EAGER="${SE_VLLM_ENFORCE_EAGER:-1}"
# Quiet down vLLM banner. Errors still surface.
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

# --- 7. Proxy (only if needed; HF_HUB_OFFLINE=1 makes it irrelevant) --------
# Compute nodes can reach the outside world only via the NHR proxy.
# We leave these *unset* by default so accidental downloads fail loudly
# rather than silently pulling from the internet during a 6-hour job.
# Set them explicitly in prefetch jobs (slurm/prefetch_models.sbatch).

# --- 8. Banner ---------------------------------------------------------------
cat <<EOF
[_common.sh] env ready
  HF_HOME              = ${HF_HOME}
  TRITON_CACHE_DIR     = ${TRITON_CACHE_DIR}
  TORCHINDUCTOR_CACHE_DIR = ${TORCHINDUCTOR_CACHE_DIR}
  VLLM_CACHE_ROOT      = ${VLLM_CACHE_ROOT}
  TMPDIR               = ${TMPDIR:-<unset>}
  SE_VLLM_ENFORCE_EAGER= ${SE_VLLM_ENFORCE_EAGER}
  OMP_NUM_THREADS      = ${OMP_NUM_THREADS:-<unset>}
  CPATH                = ${CPATH:-<unset>}
EOF
