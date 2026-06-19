#!/usr/bin/env bash
# DisTraceAI installer — single inference env ("distrace").
#
# Installs everything needed to RUN the full pipeline (generation, embeddings,
# retrieval, AND the check-worthiness detectors). The detectors run fine under
# transformers v5 alongside vLLM, so a single environment is sufficient.
#
# No CUDA compiler / no compilation: vLLM ships a prebuilt CUDA 13.0 wheel
# (matches the H200's Driver 580 / CUDA 13.0). Runs on the restricted HPC.
#
# Companion: ./setup_quantize.sh builds the AWQ weights once (needs nvcc; e.g. H200).
#
# Usage:  ./setup.sh
set -euo pipefail

VLLM_VER="0.22.1"   # supports Qwen3.5 + Gemma 4; ships torch 2.11 / CUDA 13.0 / transformers v5

module load GCC/13.2.0 2>/dev/null || true

# --- create + ACTIVATE the env, then HARD-VERIFY activation --------------------
# Running `bash setup.sh` in a non-interactive shell can leave `conda activate`
# a no-op (conda's shell function isn't loaded), which would silently install
# everything into the base/system Python. We source the hook explicitly and then
# ABORT if the distrace env is not actually active, so pip never targets base.
conda create -n distrace python=3.12 -y
eval "$(conda shell.bash hook)"
conda activate distrace

if [[ "${CONDA_DEFAULT_ENV:-}" != "distrace" ]]; then
  echo "ERROR: failed to activate the 'distrace' conda env (got '${CONDA_DEFAULT_ENV:-none}')." >&2
  echo "       Run 'conda init bash', restart your shell, then:" >&2
  echo "         conda activate distrace && ./setup.sh" >&2
  echo "       Aborting BEFORE any pip install so the base env is not polluted." >&2
  exit 1
fi
# Belt-and-braces: confirm the active python lives inside the env prefix.
PYBIN="$(python -c 'import sys; print(sys.executable)')"
if [[ "$PYBIN" != "$CONDA_PREFIX"/* ]]; then
  echo "ERROR: active python ($PYBIN) is not inside \$CONDA_PREFIX ($CONDA_PREFIX)." >&2
  echo "       Aborting to avoid polluting the base interpreter." >&2
  exit 1
fi
echo "[setup] env active: $CONDA_DEFAULT_ENV  ($PYBIN)"

# --- inference stack -----------------------------------------------------------
# vLLM brings matched torch 2.11 + CUDA 13.0 wheel + transformers>=5.
pip install "vllm==${VLLM_VER}"

# --- remaining runtime requirements --------------------------------------------
pip install -r requirements.txt

# --- NodeRAG (stock routing + compat patch; driven by in-process vLLM clients) -
bash modules/noderag/install_noderag_local.sh

echo "[setup] distrace env ready."
echo "[setup] Next (one-time, on a node with nvcc): ./setup_quantize.sh"
