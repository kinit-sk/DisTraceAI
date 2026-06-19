#!/usr/bin/env bash
# DisTraceAI — build AWQ-4bit weights for the six generators (one-time step).
#
# Run ONCE on a machine with a CUDA toolkit / nvcc (e.g. the H200 node where you
# have root). NOT needed on the restricted HPC: that node only runs ./setup.sh
# against the AWQ weights this produces in models/awq/.
#
# Uses GPTQModel (maintained successor to the stale AutoAWQ) for the AWQ algorithm
# — it supports the Gemma 4 / Qwen 3.5 architectures and emits AWQ-Marlin weights
# vLLM loads directly. Runs in the SAME "distrace" env (transformers v5), which is
# required to load gemma4 / qwen3.5 checkpoints.
#
# Usage:
#   conda activate distrace
#   ./setup_quantize.sh            # install GPTQModel, then quantize all six
#   ./setup_quantize.sh --deps     # install deps only (skip quantizing)
set -euo pipefail

# A CUDA toolkit is needed to build/run GPTQModel's quant kernels. The H200's
# CUDA 13.0 works; the HPC's CUDA 12.4 module also works (driver 580 is
# backward-compatible). Load whichever is available.
module load GCC/13.2.0   2>/dev/null || true
module load CUDA/12.4.0  2>/dev/null || true

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "ERROR: no active conda env. Run 'conda activate distrace' first." >&2
  exit 1
fi

# Pin the quantization toolchain — AWQ output weights are sensitive to the
# GPTQModel + transformers versions, so pinning makes the weights reproducible.
GPTQMODEL_VER="5.8.0"     # adds Qwen3.5 + Gemma4 + AWQ-Marlin; verify availability
echo "[quantize] installing GPTQModel==${GPTQMODEL_VER} + datasets (needs nvcc)…"
INSTALL_KERNELS=1 pip install "gptqmodel==${GPTQMODEL_VER}" datasets

if [[ "${1:-}" == "--deps" ]]; then
  echo "[quantize] deps installed; skipping quantization (--deps)."
  exit 0
fi

echo "[quantize] building AWQ-4bit weights for all six generators…"
python quantize.py --all --out models/awq

echo "[quantize] done. AWQ weights in models/awq/. Context-1 is NOT self-quantized"
echo "           (its low-bit checkpoint is QAT-distilled upstream — use a"
echo "           community MXFP4/4-bit checkpoint or run bf16 on a large GPU)."
