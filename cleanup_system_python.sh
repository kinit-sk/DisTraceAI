#!/usr/bin/env bash
# cleanup_system_python.sh
# Reverts the damage from running `bash setup.sh` WITHOUT an activated conda env:
# the pip installs landed in the system (base) Python instead of the distrace env.
# This uninstalls exactly the packages setup.sh installs, from whichever python is
# active when you run it. Run it with the SAME interpreter that got polluted
# (i.e. plain `bash cleanup_system_python.sh`, no conda env active).
#
# It is conservative: it only removes the known DisTraceAI deps + their heavy
# transitive ones that vllm drags in, and asks for confirmation first.
set -euo pipefail

PY="${PYTHON:-python3}"
echo ">> Target interpreter: $($PY -c 'import sys; print(sys.executable)')"
echo ">> This will pip-uninstall DisTraceAI packages from the above interpreter."
read -r -p ">> Proceed? [y/N] " ans
[[ "$ans" == "y" || "$ans" == "Y" ]] || { echo "aborted."; exit 0; }

# Packages setup.sh installs directly. vllm pulls a large dependency tree
# (torch, transformers, xformers, ray, etc.); pip-autoremove would be ideal but
# isn't standard, so we uninstall vllm + the direct requirements and the biggest
# transitive offenders explicitly.
PKGS=(
  vllm
  NodeRAG
  chromadb
  numpy pandas scikit-learn rank-bm25
  feedparser trafilatura fasttext-langdetect bertopic
  rich pyyaml pytest
  transformers tokenizers safetensors
  torch torchvision torchaudio
  xformers ray sentencepiece
  gptqmodel datasets
)

echo ">> Uninstalling ${#PKGS[@]} packages (ignoring any not present)…"
$PY -m pip uninstall -y "${PKGS[@]}" 2>/dev/null || true

echo ""
echo ">> Done. Note: pip does not remove sub-dependencies automatically, so a few"
echo ">>  small transitive packages may remain. To see what's left:"
echo ">>     $PY -m pip list"
echo ">>  For a fully clean base, consider recreating the base env, but the above"
echo ">>  removes everything heavy that setup.sh introduced."
echo ""
echo ">> Then install correctly INTO the conda env:"
echo ">>     conda activate distrace && ./setup.sh"
