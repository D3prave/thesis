#!/bin/bash -l
#SBATCH --job-name=se-prefetch
#SBATCH --partition=tiny
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/prefetch_%j.out
#SBATCH --error=logs/prefetch_%j.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
# ============================================================================
# Pre-fetch all model weights to the vault HF cache before submitting GPU jobs.
#
# This is a CPU-only job — no GPU required.  Run it once; after completion
# all compute scripts (phase1_7b, phase1_70b, phase2_nli, …) will find their
# models cached and will not attempt any network access at runtime.
#
# Usage:
#   mkdir -p logs
#   sbatch slurm/prefetch_models.sbatch
#
# To also download gated models (Llama-3.1-70B) you must pass your HF token:
#   sbatch --export=ALL,HF_TOKEN=hf_... slurm/prefetch_models.sbatch
#
# Alternatively, run scripts/download_models.sh interactively on a login node.
# ============================================================================

set -euo pipefail

echo "=== Model prefetch job ${SLURM_JOB_ID} on $(hostname) ==="
echo "Start time: $(date --iso-8601=seconds)"

# ---------- Environment ----------
module load python

REPO_ROOT="${SLURM_SUBMIT_DIR}"
source "${REPO_ROOT}/.venv/bin/activate"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# Load .env from repo root if present (never commit .env — it's in .gitignore)
# shellcheck disable=SC1091
[ -f "${REPO_ROOT}/.env" ] && source "${REPO_ROOT}/.env"

# NHR@FAU proxy — required for outbound internet on compute nodes
export http_proxy="${http_proxy:-http://proxy.nhr.fau.de:80}"
export https_proxy="${https_proxy:-http://proxy.nhr.fau.de:80}"

# Always write to vault — same location as all compute scripts
export HF_HOME="/home/vault/b192aa/b192aa36/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}"

# Set HF_TOKEN if provided via --export (enables gated models like Llama)
if [ -n "${HF_TOKEN:-}" ]; then
    export HF_TOKEN
    echo "HF_TOKEN provided — gated models (Llama etc.) will be accessible."
else
    echo "Warning: HF_TOKEN not set. Gated models (Llama-3.1-70B) will fail."
    echo "         Pass it with:  sbatch --export=ALL,HF_TOKEN=hf_... $0"
fi

# ---------- Models to download ----------
# Edit this list to add or remove models.
MODELS_7B=(
    "mistralai/Mistral-7B-Instruct-v0.3"
)
MODELS_NLI=(
    "cross-encoder/nli-deberta-v3-base"
    "sentence-transformers/all-MiniLM-L6-v2"   # KLE embedding model
)
MODELS_70B=(
    # Gated — requires HF_TOKEN and accepted licence on huggingface.co
    "meta-llama/Llama-3.1-70B-Instruct"
)

download_model() {
    local model="$1"
    echo ""
    echo "--- Downloading: ${model} ---"
    python - <<PYEOF
import sys
from huggingface_hub import snapshot_download
import os

model = "${model}"
hf_hub_cache = os.environ.get("HF_HUB_CACHE", "")

try:
    path = snapshot_download(
        repo_id=model,
        cache_dir=hf_hub_cache,
        # ignore safetensors vs bin — grab everything needed for inference
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*"],
    )
    print(f"OK: {model} -> {path}")
except Exception as e:
    print(f"FAILED: {model}: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
}

# ---------- Download ----------
echo ""
echo "=== Downloading 7B / 8B generation models ==="
for MODEL in "${MODELS_7B[@]}"; do
    download_model "${MODEL}"
done

echo ""
echo "=== Downloading NLI entailment models ==="
for MODEL in "${MODELS_NLI[@]}"; do
    download_model "${MODEL}"
done

if [ -n "${HF_TOKEN:-}" ]; then
    echo ""
    echo "=== Downloading 70B generation models (gated) ==="
    for MODEL in "${MODELS_70B[@]}"; do
        download_model "${MODEL}"
    done
else
    echo ""
    echo "Skipping 70B models (no HF_TOKEN). Re-run with token when needed."
fi

# ---------- Summary ----------
echo ""
echo "=== Cache contents ==="
du -sh "${HF_HUB_CACHE}" 2>/dev/null || true
ls "${HF_HUB_CACHE}" 2>/dev/null | grep '^models--' || true

echo ""
echo "=== Prefetch complete ==="
echo "End time: $(date --iso-8601=seconds)"
echo "You can now submit GPU jobs:"
echo "  sbatch --export=ALL,DATA_PATH=data/processed/triviaqa_val_500.jsonl slurm/phase1_7b_alex.sbatch"
