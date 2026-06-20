#!/usr/bin/env bash
# DisTraceAI installer — single inference env ("distrace").
#
# ONE command does everything: creates the env, installs the inference stack
# (vLLM + torch + transformers v5), the pipeline requirements, NodeRAG (patched
# local clone), removes the kernels package, and finally VERIFIES every critical
# import — reinstalling anything missing.
#
# Robustness notes (learned the hard way):
#   * NO `set -e`  — a single failed install (network blip on a multi-GB wheel,
#     one unresolvable line) must not abort the run and skip later steps.
#   * NO `set -u`  — conda's activate/install shell functions reference unbound
#     variables; under nounset a NON-interactive shell EXITS on the first one,
#     which silently killed setup right after the CUDA step.
#   * NO `2>/dev/null` on conda — hiding stderr is how that silent death went
#     unnoticed. Everything is visible now.
# The only hard-abort is failure to activate the env (so we never touch base).
#
# Usage:  bash setup.sh
set -o pipefail

VLLM_VER="0.22.1"   # supports Qwen3.5 + Gemma 4; ships torch 2.11 / CUDA 13.0 / transformers v5
export VLLM_VER

_banner() { echo; echo "========================================================"; \
            echo ">> $*"; echo "========================================================"; }

module load GCC/13.2.0 2>/dev/null || true

# --- create + ACTIVATE the env, then HARD-VERIFY activation (only hard abort) --
_banner "Creating + activating conda env 'distrace'"
conda create -n distrace python=3.12 -y || true
eval "$(conda shell.bash hook)"
conda activate distrace || true

if [[ "${CONDA_DEFAULT_ENV:-}" != "distrace" ]]; then
  echo "ERROR: failed to activate the 'distrace' conda env (got '${CONDA_DEFAULT_ENV:-none}')." >&2
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

# --- CUDA runtime / libcuda visibility (best-effort, never fatal) --------------
# nvcc + cudart for any FlashInfer JIT path; harmless if the node uses module
# CUDA instead. Run in a subshell so its activation side-effects can't change
# our shell, and never let it stop the script. (stderr is intentionally visible.)
_banner "CUDA toolkit + libcuda visibility (best-effort)"
( conda install -n distrace -c nvidia cuda-toolkit=13.3 -y ) \
  && echo "[setup] cuda-toolkit installed." \
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

# --- inference stack -----------------------------------------------------------
_banner "Installing vLLM ${VLLM_VER} (brings torch 2.11 / CUDA 13.0 / transformers v5)"
python -m pip install "vllm==${VLLM_VER}" \
  || echo "[setup] WARNING: vLLM install returned non-zero — final verification will retry it."

# --- pipeline requirements -----------------------------------------------------
_banner "Installing pipeline requirements"
python -m pip install -r requirements.txt \
  || echo "[setup] WARNING: requirements install returned non-zero — self-heal will fix gaps."

# --- NodeRAG (patched local clone; driven by in-process vLLM clients) ----------
# After the main stack, so NodeRAG only ADDS missing deps (installed unpinned)
# and cannot downgrade vLLM's stack. Never let a hiccup here stop verification.
_banner "Installing NodeRAG (patched local clone)"
bash modules/noderag/install_noderag_local.sh \
  || echo "[setup] WARNING: NodeRAG install returned non-zero — see its output above."

# --- transformers 5.12 + kernels 0.15 import crash workaround ------------------
_banner "Removing 'kernels' (transformers v5 import crash workaround)"
python -m pip uninstall -y kernels 2>/dev/null || true

# --- FINAL self-heal + verification (single source of truth) -------------------
_banner "Verifying the environment"
python - <<'PYCHECK'
import importlib, subprocess, sys, os
CRITICAL = {  # import_name : pip_spec ('' = do not pip-install; vllm-managed or editable)
    "vllm":        "vllm==%s" % os.environ.get("VLLM_VER", "0.22.1"),
    "torch":       "",
    "transformers":"",
    "rich":        "rich",
    "pandas":      "pandas",
    "numpy":       "numpy",
    "yaml":        "pyyaml",
    "sklearn":     "scikit-learn",
    "rank_bm25":   "rank-bm25",
    "feedparser":  "feedparser",
    "trafilatura": "trafilatura",
    "chromadb":    "chromadb",
    "hnswlib":     "hnswlib",
    "NodeRAG":     "",
}
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
print("[setup] === environment verification ===")
for m in CRITICAL:
    print(f"[setup]   {'OK     ' if importable(m) else 'MISSING'}  {m}")
still = [m for m in CRITICAL if not importable(m)]
if still:
    print("[setup] WARNING: still not importable: " + ", ".join(still), file=sys.stderr)
    if "NodeRAG" in still:
        print("[setup]   → re-run: bash modules/noderag/install_noderag_local.sh", file=sys.stderr)
    sys.exit(1)
print("[setup] All critical packages import OK.")
PYCHECK
rc=$?

_banner "distrace env ready"
[[ $rc -ne 0 ]] && echo "[setup] (some packages still missing — see WARNING above)"
exit $rc
