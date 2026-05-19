#!/usr/bin/env bash
# =============================================================================
# download_models.sh — pre-fetch all model weights to vault before GPU jobs.
#
# Run this INTERACTIVELY on a login node (no GPU needed, login nodes have
# internet access):
#
#   bash scripts/download_models.sh
#
# To download gated models (Llama-3.1-70B), pass your HF token:
#
#   HF_TOKEN=hf_... bash scripts/download_models.sh
#
# Or set it permanently in your ~/.bashrc:
#   export HF_TOKEN=hf_...
#
# After this script completes, sbatch the SLURM jobs as normal — they will
# find their models in cache and skip all network access.
# =============================================================================

set -euo pipefail

# ---------- Config ----------
# Load .env from repo root if present (never commit .env — it's in .gitignore)
REPO_ROOT_EARLY="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
[ -f "${REPO_ROOT_EARLY}/.env" ] && source "${REPO_ROOT_EARLY}/.env"

# NHR@FAU proxy — required for outbound internet on both login and compute nodes
export http_proxy="${http_proxy:-http://proxy.nhr.fau.de:80}"
export https_proxy="${https_proxy:-http://proxy.nhr.fau.de:80}"

# Explicitly clear offline flags — download scripts must never run in offline mode
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE
unset HF_DATASETS_OFFLINE

HF_HOME="${HF_HOME:-/home/vault/b192aa/b192aa36/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}"

export HF_HOME HF_HUB_CACHE TRANSFORMERS_CACHE="${HF_HUB_CACHE}"

# HF_TOKEN: use env var if set, otherwise prompt
if [ -z "${HF_TOKEN:-}" ]; then
    echo ""
    echo "HF_TOKEN is not set."
    echo "You need it only for gated models (Llama-3.1-70B)."
    read -r -p "Enter HF token (or press Enter to skip gated models): " HF_TOKEN
fi
[ -n "${HF_TOKEN:-}" ] && export HF_TOKEN && echo "HF_TOKEN set."

echo ""
echo "Cache location: ${HF_HUB_CACHE}"
echo ""

# ---------- Activate venv ----------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${REPO_ROOT}/.venv"

if [ ! -f "${VENV}/bin/activate" ]; then
    echo "ERROR: venv not found at ${VENV}"
    echo "       Run: cd ${REPO_ROOT} && uv sync"
    exit 1
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# ---------- Download helper ----------
download_model() {
    local model="$1"
    local cache_dir="${HF_HUB_CACHE}/models--$(echo "${model}" | sed 's|/|--|g')"

    if [ -d "${cache_dir}" ]; then
        echo "  SKIP (already cached): ${model}"
        echo "       ${cache_dir}"
        return 0
    fi

    echo "  Downloading: ${model}"
    python - <<PYEOF
import sys
from huggingface_hub import snapshot_download
import os

model = "${model}"
hf_hub_cache = os.environ["HF_HUB_CACHE"]

try:
    path = snapshot_download(
        repo_id=model,
        cache_dir=hf_hub_cache,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*"],
    )
    print(f"  OK: {model}")
    print(f"      -> {path}")
except Exception as e:
    print(f"  FAILED: {model}: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
}

# ---------- Models ----------
echo "=== 7B / 8B generation models ==="
download_model "mistralai/Mistral-7B-Instruct-v0.3"

echo ""
echo "=== NLI entailment + KLE embedding models ==="
download_model "cross-encoder/nli-deberta-v3-base"
download_model "sentence-transformers/all-MiniLM-L6-v2"

echo ""
echo "=== 70B generation models (gated — requires accepted licence + HF_TOKEN) ==="
if [ -n "${HF_TOKEN:-}" ]; then
    download_model "meta-llama/Llama-3.1-70B-Instruct"
else
    echo "  SKIP: HF_TOKEN not set. Re-run with HF_TOKEN=hf_... bash $0"
fi

# ---------- Summary ----------
echo ""
echo "=== Cache summary ==="
du -sh "${HF_HUB_CACHE}" 2>/dev/null && echo ""
ls "${HF_HUB_CACHE}" 2>/dev/null | grep '^models--' | sed 's/^/  /' || true

echo ""
echo "All done. You can now submit GPU jobs:"
echo "  sbatch --export=ALL,DATA_PATH=data/processed/triviaqa_val_500.jsonl \\"
echo "      slurm/phase1_7b_alex.sbatch"
