#!/usr/bin/env bash
# install_noderag_local.sh
# Clone the pinned NodeRAG, apply the DisTraceAI COMPATIBILITY patch (local-model
# token counting + small-graph guards — NOT a model provider), and install it
# editable into the active conda env.
#
# NodeRAG's model routing is left STOCK (SpecFi-faithful): the pipeline injects
# its own in-process vLLM clients at runtime, so NodeRAG needs no provider of its
# own and llama-cpp is not involved anywhere.
#
# Usage:
#   conda activate distrace
#   ./install_noderag_local.sh [SRC_DIR]
set -euo pipefail

PIN="f77dd6adb34cf4dda1d88b30b2bf0b17d14480a9"
REPO="https://github.com/Terry-Xu-666/NodeRAG.git"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$HERE/noderag_compat.patch"
SRC_DIR="${1:-$HERE/NodeRAG_local}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "ERROR: no active conda env. Run 'conda activate distrace' first." >&2
  exit 1
fi
echo ">> Target conda env: $CONDA_PREFIX"
echo ">> NodeRAG source dir: $SRC_DIR"

# 1) clone + reset to a pristine pin so the patch always applies cleanly
if [[ ! -d "$SRC_DIR/.git" ]]; then
  git clone "$REPO" "$SRC_DIR"
fi
cd "$SRC_DIR"
git fetch --all --tags >/dev/null 2>&1 || true
git reset --hard "$PIN"
git clean -fd >/dev/null 2>&1 || true
echo ">> Reset NodeRAG to pristine @ $PIN"

# 2) apply the compatibility patch (clean tree -> always a fresh forward apply)
git apply --check "$PATCH"
git apply "$PATCH"
echo ">> Applied DisTraceAI compat patch (token_utils local-model fallback +"
echo "   small-graph guards in attribute/summary/relationship generation)."

# 3) install the patched NodeRAG (editable) into the active env
pip install -e .

echo ""
echo ">> Done. NodeRAG installed (stock model routing + compat patch)."
echo ">> The pipeline drives NodeRAG via in-process vLLM clients at runtime."
