#!/usr/bin/env bash
# install_noderag_local.sh
# Clone NodeRAG, apply the DisTraceAI llama.cpp patch (replaces the commercial
# OpenAI/Gemini model integration with a local llama-cpp-python integration), and
# install it editable into the ACTIVE conda environment.
#
# Usage:
#   conda activate distrace          # the env DisTraceAI runs in
#   ./install_noderag_local.sh [SRC_DIR]
#
# SRC_DIR defaults to NodeRAG_local/ alongside this script (under modules/noderag/).
set -euo pipefail

PIN="f77dd6adb34cf4dda1d88b30b2bf0b17d14480a9"
REPO="https://github.com/Terry-Xu-666/NodeRAG.git"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$HERE/noderag_llamacpp.patch"
SRC_DIR="${1:-$HERE/NodeRAG_local}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "ERROR: no active conda env. Run 'conda activate <env>' first." >&2
  exit 1
fi
echo ">> Target conda env: $CONDA_PREFIX"
echo ">> NodeRAG source dir: $SRC_DIR"

# 1) clone (full clone so the pinned commit is reachable) + pin, then reset the
#    working tree to a pristine pin so the patch always applies cleanly on re-runs
if [[ ! -d "$SRC_DIR/.git" ]]; then
  git clone "$REPO" "$SRC_DIR"
fi
cd "$SRC_DIR"
git fetch --all --tags >/dev/null 2>&1 || true
git reset --hard "$PIN"
git clean -fd >/dev/null 2>&1 || true
echo ">> Reset NodeRAG to pristine @ $PIN"

# 2) apply the patch (clean tree → always a fresh forward apply)
git apply --check "$PATCH"
git apply "$PATCH"
echo ">> Applied llama.cpp + small-graph patch (LLM.py, LLM_route.py, token_utils.py, attribute_generation.py, summary_generation.py)."

# 3) llama-cpp-python — build with CUDA when nvcc is present, else CPU
if python -c "import llama_cpp" >/dev/null 2>&1; then
  echo ">> llama-cpp-python already installed."
elif command -v nvcc >/dev/null 2>&1; then
  echo ">> Building llama-cpp-python with CUDA…"
  CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-cache-dir llama-cpp-python
else
  echo ">> nvcc not found — building CPU llama-cpp-python…"
  pip install --no-cache-dir llama-cpp-python
fi

# 4) install the patched NodeRAG (editable) into the active env
pip install -e .

echo ""
echo ">> Done. Sanity check:"
python -c "from NodeRAG.LLM.LLM_route import Llama_cpp, Llama_cpp_Embedding; print('llama.cpp providers available')"
echo ">> Configure NodeRAG with service_provider: llama_cpp  (see README.md / the example YAML)."
