#!/usr/bin/env bash
# DisTraceAI installer — backend-selectable.
#
# DisTraceAI ships TWO inference backends; you pick which one to install.
# Each backend lives in its OWN conda env so they can coexist on disk:
#
#     distrace-vllm    →  vLLM 0.22.1   (requires CUDA 13.x, Python 3.12)
#     distrace-llama   →  llama-cpp     (requires CUDA 12.x, Python 3.11)
#
# At runtime the active backend is selected via the TUI Settings menu (or
# DISTRACE_LLM_BACKEND env var) and the matching env is `conda activate`d by
# `./activate_distrace.sh` — see that helper for one-shot launch.
#
# Both branches install:
#   * the chosen inference stack
#   * the shared pipeline requirements (requirements.txt)
#   * NodeRAG via the patched local clone (needed for specfi-cs / specfi-ccs)
#
# Robustness notes (learned the hard way):
#   * NO `set -e` — one failed dep (network blip on a multi-GB wheel) must not
#     abort the run and skip later steps. The verification pass at the end
#     self-heals anything that didn't install cleanly.
#   * NO `set -u` — conda's activate/install shell functions reference unbound
#     variables; nounset kills setup right after the CUDA step.
#   * NO `2>/dev/null` on conda — hiding stderr is how the silent death goes
#     unnoticed. Everything stays visible.
#
# Usage:  bash setup.sh         (interactive, prompts for backend)
#         bash setup.sh vllm    (non-interactive)
#         bash setup.sh llama   (non-interactive)
set -o pipefail

VLLM_VER="0.22.1"     # ships torch 2.11 / CUDA 13.0 / transformers v5
VLLM_ENV="distrace-vllm"
VLLM_PY="3.12"
LLAMA_ENV="distrace-llama"
LLAMA_PY="3.11"

_banner() { echo; echo "========================================================"; \
            echo ">> $*"; echo "========================================================"; }

# ---------------------------------------------------------------------------
# 1) Show CUDA requirements first; print-and-warn on a host CUDA mismatch.
#    We detect via nvcc / nvidia-smi (best effort — HPC users often have
#    multiple CUDA modules loadable, so we never refuse to proceed).
# ---------------------------------------------------------------------------
_banner "DisTraceAI installer — backend selection"
cat <<'EOF'
DisTraceAI provides two LLM inference backends. Pick ONE to install:

   Backend     CUDA required   Conda env             Python
   ─────────   ─────────────   ───────────────────   ──────
   vllm        13.x (13.3 OK)  distrace-vllm         3.12
   llama-cpp   12.x (12.4 OK)  distrace-llama        3.11

The other backend can be installed later by re-running this script.
EOF

_detect_cuda() {
  if command -v nvcc >/dev/null 2>&1; then
    nvcc --version | awk '/release/ {print $5}' | tr -d ','
  elif command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi | awk -F'CUDA Version: ' '/CUDA Version:/ {print $2}' | awk '{print $1}'
  else
    echo ""
  fi
}
HOST_CUDA="$(_detect_cuda)"
if [[ -n "$HOST_CUDA" ]]; then
  echo
  echo "[setup] detected host CUDA toolkit / driver: $HOST_CUDA"
else
  echo
  echo "[setup] (could not detect host CUDA toolkit — nvcc / nvidia-smi unavailable)"
fi

# ---------------------------------------------------------------------------
# 2) Backend choice — CLI arg first, then interactive prompt.
# ---------------------------------------------------------------------------
CHOICE="${1:-}"
case "$CHOICE" in
  vllm|VLLM)         BACKEND=vllm  ;;
  llama|llama-cpp|llamacpp|LLAMA)  BACKEND=llama ;;
  "")
    echo
    echo "Which backend do you want to install?"
    echo "  [1] vllm        (CUDA 13.x — fastest on H100/H200 / A100)"
    echo "  [2] llama-cpp   (CUDA 12.x — works on older toolchains, GGUF quants)"
    while true; do
      read -r -p "Choose 1 or 2: " sel
      case "$sel" in
        1|vllm)  BACKEND=vllm; break ;;
        2|llama|llama-cpp) BACKEND=llama; break ;;
        *) echo "  (please type 1 or 2)" ;;
      esac
    done
    ;;
  *)
    echo "[setup] ERROR: unknown backend '$CHOICE'. Use 'vllm' or 'llama'." >&2
    exit 1
    ;;
esac

# Warn (don't refuse) on an obvious CUDA mismatch.
if [[ -n "$HOST_CUDA" ]]; then
  HOST_CUDA_MAJOR="${HOST_CUDA%%.*}"
  if [[ "$BACKEND" == "vllm"  && "$HOST_CUDA_MAJOR" != "13" ]]; then
    echo "[setup] WARNING: vLLM expects CUDA 13.x but host CUDA is $HOST_CUDA."
    echo "                 The install may still succeed if the right CUDA module is loaded."
  fi
  if [[ "$BACKEND" == "llama" && "$HOST_CUDA_MAJOR" != "12" ]]; then
    echo "[setup] WARNING: llama-cpp expects CUDA 12.x but host CUDA is $HOST_CUDA."
    echo "                 The install may still succeed if the right CUDA module is loaded."
  fi
fi

# ---------------------------------------------------------------------------
# 3) Create + ACTIVATE the chosen env. Only HARD-ABORT case is activation failure
#    — that's the line between a clean install and polluting the base env.
# ---------------------------------------------------------------------------
if [[ "$BACKEND" == "vllm" ]]; then
  ENV_NAME="$VLLM_ENV"; PY_VER="$VLLM_PY"
else
  ENV_NAME="$LLAMA_ENV"; PY_VER="$LLAMA_PY"
fi

_banner "Creating + activating conda env '$ENV_NAME' (python $PY_VER)"
conda create -n "$ENV_NAME" "python=$PY_VER" -y || true
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME" || true

if [[ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]]; then
  echo "ERROR: failed to activate the '$ENV_NAME' conda env (got '${CONDA_DEFAULT_ENV:-none}')." >&2
  echo "       Run 'conda init bash', restart your shell, then:  bash setup.sh" >&2
  echo "       Aborting BEFORE any pip install so the base env is not polluted." >&2
  exit 1
fi
PYBIN="$(python -c 'import sys; print(sys.executable)')"
case "$PYBIN" in
  "$CONDA_PREFIX"/*) : ;;
  *) echo "ERROR: active python ($PYBIN) is not inside \$CONDA_PREFIX ($CONDA_PREFIX). Aborting." >&2
     exit 1 ;;
esac
echo "[setup] env active: $CONDA_DEFAULT_ENV  ($PYBIN)"

# ---------------------------------------------------------------------------
# 4) Backend-specific installation branch
# ---------------------------------------------------------------------------
if [[ "$BACKEND" == "vllm" ]]; then
  # ---- vLLM branch -------------------------------------------------------
  module load GCC/13.2.0 2>/dev/null || true

  _banner "CUDA toolkit + libcuda visibility (best-effort)"
  # nvcc + cudart for any FlashInfer JIT path; harmless if the node uses module
  # CUDA instead. Run in a subshell so its activation side-effects can't change
  # our shell, and never let it stop the script.
  ( conda install -n "$ENV_NAME" -c nvidia cuda-toolkit=13.3 -y ) \
    && echo "[setup] cuda-toolkit 13.3 installed." \
    || echo "[setup] (cuda-toolkit step returned non-zero — using module/system CUDA; continuing)"

  _SYS_LIB="/usr/lib/x86_64-linux-gnu"
  if [[ -e "$_SYS_LIB/libcuda.so" || -e "$_SYS_LIB/libcuda.so.1" ]]; then
    export LD_LIBRARY_PATH="$_SYS_LIB:${LD_LIBRARY_PATH:-}"
    export LIBRARY_PATH="$_SYS_LIB:${LIBRARY_PATH:-}"
    export LDFLAGS="-L$_SYS_LIB ${LDFLAGS:-}"
    echo "[setup] libcuda found in $_SYS_LIB — exported LD_LIBRARY_PATH/LIBRARY_PATH/LDFLAGS (this shell only)."
  else
    echo "[setup] (no libcuda.so in $_SYS_LIB — relying on module/conda CUDA)"
  fi
  rm -rf "$HOME/.cache/flashinfer" 2>/dev/null || true   # avoid stale JIT cache

  _banner "Installing vLLM ${VLLM_VER} (brings torch 2.11 / CUDA 13.0 / transformers v5)"
  export VLLM_VER
  python -m pip install "vllm==${VLLM_VER}" \
    || echo "[setup] WARNING: vLLM install returned non-zero — final verification will retry it."

else
  # ---- llama-cpp branch --------------------------------------------------
  _banner "Loading CUDA 12.4 + GCC + CMake modules (HPC)"
  module purge 2>/dev/null || true
  module load GCC/13.2.0 2>/dev/null || true
  module load CUDA/12.4.0 2>/dev/null || true
  module load CMake/3.27.6 2>/dev/null || true

  # Build env for llama-cpp-python's CMake build. We deliberately point the
  # CUDA host compiler at the SYSTEM g++ (not the GCC/13.2 just loaded) because
  # CUDA 12.4's nvcc rejects very recent GCC versions; the system g++ is older
  # and known-good.
  export CC=/usr/bin/gcc
  export CXX=/usr/bin/g++
  export CUDA_HOME="${CUDA_ROOT:-/usr/local/cuda}"
  export CUDACXX="$CUDA_HOME/bin/nvcc"
  export CUDAHOSTCXX=/usr/bin/g++
  export CMAKE_ARGS="-DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON"
  export FORCE_CMAKE=1
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

  _banner "Cloning + building llama-cpp-python (CUDA backend, modules/llama-cpp)"
  mkdir -p modules
  if [[ ! -d modules/llama-cpp-python ]]; then
    git clone --recurse-submodules https://github.com/abetlen/llama-cpp-python.git \
        modules/llama-cpp-python \
      || echo "[setup] WARNING: llama-cpp-python clone failed — final verification will report it."
  else
    echo "[setup] modules/llama-cpp-python already exists — skipping clone."
  fi
  if [[ -d modules/llama-cpp-python ]]; then
    ( cd modules/llama-cpp-python && pip install --no-cache-dir --force-reinstall . ) \
      || echo "[setup] WARNING: llama-cpp-python build returned non-zero — see output above."
  fi
fi

# ---------------------------------------------------------------------------
# 5) Shared steps for both backends
# ---------------------------------------------------------------------------

_banner "Installing pipeline requirements (requirements.txt)"
python -m pip install -r requirements.txt \
  || echo "[setup] WARNING: requirements install returned non-zero — self-heal will fix gaps."

if [[ "$BACKEND" == "llama" ]]; then
  # llama-cpp pins numpy 1.x and a known-good torch wheel for CUDA 12.4
  _banner "Pinning numpy + torch for the llama-cpp env"
  python -m pip install numpy==1.26.4 \
    || echo "[setup] WARNING: numpy 1.26.4 pin returned non-zero — see output above."
  python -m pip install --upgrade \
      "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1" \
      --index-url https://download.pytorch.org/whl/cu124 \
    || echo "[setup] WARNING: torch cu124 install returned non-zero — see output above."
fi

# ---- NodeRAG (patched local clone) — needed for specfi-cs / specfi-ccs ----
# Installed AFTER the main stack so NodeRAG only ADDS missing deps (installed
# unpinned) and cannot downgrade the vLLM stack or the llama-cpp pins.
_banner "Installing NodeRAG (patched local clone) — both backends use it for SpecFi"
if [[ -x modules/noderag/install_noderag_local.sh ]]; then
  bash modules/noderag/install_noderag_local.sh \
    || echo "[setup] WARNING: NodeRAG install returned non-zero — see its output above."
else
  echo "[setup] WARNING: modules/noderag/install_noderag_local.sh not found or not executable;"
  echo "                 SpecFi methods (specfi-cs / specfi-ccs) will not be available."
fi

if [[ "$BACKEND" == "vllm" ]]; then
  # transformers 5.12 + kernels 0.15 import crash workaround — applies only
  # to the vLLM stack (transformers v5).
  _banner "Removing 'kernels' (transformers v5 import crash workaround)"
  python -m pip uninstall -y kernels 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# 6) FINAL self-heal + verification (single source of truth)
# ---------------------------------------------------------------------------
_banner "Verifying the environment"
export DISTRACE_VERIFY_BACKEND="$BACKEND"
python - <<'PYCHECK'
import importlib, subprocess, sys, os

backend = os.environ.get("DISTRACE_VERIFY_BACKEND", "vllm")

# Imports common to BOTH backends. (NodeRAG is installed for both; SpecFi
# requires it.)
COMMON = {
    "torch":        "",
    "transformers": "",
    "rich":         "rich",
    "pandas":       "pandas",
    "numpy":        "numpy",
    "yaml":         "pyyaml",
    "sklearn":      "scikit-learn",
    "rank_bm25":    "rank-bm25",
    "feedparser":   "feedparser",
    "trafilatura":  "trafilatura",
    "chromadb":     "chromadb",
    "duckduckgo_search": "duckduckgo-search",
    "NodeRAG":      "",
    # narwhals.stable.v2 is needed by modern sklearn; an older narwhals (1.x)
    # imports as `narwhals` but lacks `stable.v2` and crashes sklearn at import.
    # Listing the dotted name forces the verifier to load `stable.v2` itself.
    "narwhals.stable.v2": "narwhals>=2.0",
}
if backend == "vllm":
    BACKEND_PKGS = {
        "vllm":     "vllm==%s" % os.environ.get("VLLM_VER", "0.22.1"),
        "hnswlib":  "hnswlib",
    }
else:
    BACKEND_PKGS = {
        "llama_cpp": "",   # built from local source — no pip self-heal possible
        "sentence_transformers": "sentence-transformers",
    }
CRITICAL = {**COMMON, **BACKEND_PKGS}

def importable(mod):
    try:
        importlib.invalidate_caches(); importlib.import_module(mod); return True
    except Exception:
        return False

for mod in [m for m in CRITICAL if not importable(m)]:
    spec = CRITICAL[mod]
    if spec:
        print(f"[setup]   reinstalling {mod}  ({spec}) …")
        subprocess.call([sys.executable, "-m", "pip", "install", spec])

print()
print(f"[setup] === environment verification (backend={backend}) ===")
for m in CRITICAL:
    print(f"[setup]   {'OK     ' if importable(m) else 'MISSING'}  {m}")
still = [m for m in CRITICAL if not importable(m)]
if still:
    print("[setup] WARNING: still not importable: " + ", ".join(still), file=sys.stderr)
    if "NodeRAG" in still:
        print("[setup]   → re-run: bash modules/noderag/install_noderag_local.sh", file=sys.stderr)
    if "llama_cpp" in still:
        print("[setup]   → re-run: (cd modules/llama-cpp-python && pip install --no-cache-dir --force-reinstall .)",
              file=sys.stderr)
    sys.exit(1)
print(f"[setup] All critical packages import OK for backend={backend}.")
PYCHECK
rc=$?

# ---------------------------------------------------------------------------
# 7) Persist the chosen backend into config.json so the TUI starts in the right
#    mode and ./activate_distrace.sh can `conda activate` the matching env.
# ---------------------------------------------------------------------------
_banner "Persisting llm_backend=$BACKEND into config.json"
python - <<PYSAVE
import json, os
from pathlib import Path
cfg_path = Path("config.json")
data = {}
if cfg_path.exists():
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        data = {}
data["llm_backend"] = ("vllm" if "$BACKEND" == "vllm" else "llama-cpp")
cfg_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
print(f"[setup] config.json updated:  llm_backend = {data['llm_backend']}")
PYSAVE

_banner "distrace env '$ENV_NAME' ready"
if [[ $rc -ne 0 ]]; then
  echo "[setup] (some packages still missing — see WARNING above)"
fi
cat <<EOF

Next step:
    # Run pipeline from one shell with auto-activation:
    ./activate_distrace.sh         # picks up llm_backend from config.json

    # Or activate manually:
    conda activate $ENV_NAME
    python main.py

Switch backends later by re-running:  bash setup.sh
EOF
exit $rc
