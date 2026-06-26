#!/bin/bash
# DisTraceAI dependency installer.
#
# Usage:
#   ./setup.sh hpc     # HPC cluster: module-based toolchain
#   ./setup.sh desktop # Local workstation with an NVIDIA GPU
#   ./setup.sh cpu     # CPU-only install (no CUDA)
#
# Each mode will ask whether to install the vLLM or llama-cpp backend and
# create the appropriate conda environment:
#   distrace-vllm   — for the vLLM backend  (bf16, full HuggingFace repos)
#   distrace-llama  — for the llama-cpp backend  (GGUF quant models)
#
# Both envs install the shared requirements.txt.  Only one is active at a time;
# switch with:  conda activate distrace-vllm  or  conda activate distrace-llama
set -euo pipefail

MODE="${1:-}"
case "$MODE" in
    hpc|desktop|cpu) ;;
    *)
        echo "Usage: $0 {hpc|desktop|cpu}" >&2
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------
echo ""
echo "┌─────────────────────────────────────────────────────┐"
echo "│  DisTraceAI — LLM backend selection                │"
echo "├─────────────────────────────────────────────────────┤"
echo "│  1) vLLM      — bf16 via HuggingFace (recommended) │"
echo "│  2) llama-cpp — GGUF quantised models              │"
echo "└─────────────────────────────────────────────────────┘"
echo ""

while true; do
    read -rp "Select backend [1/2]: " BACKEND_CHOICE
    case "$BACKEND_CHOICE" in
        1) BACKEND="vllm";      ENV_NAME="distrace-vllm";  break ;;
        2) BACKEND="llama-cpp"; ENV_NAME="distrace-llama"; break ;;
        *) echo "Please enter 1 or 2." ;;
    esac
done

echo ""
echo "[setup] Mode: ${MODE}  |  Backend: ${BACKEND}  |  Env: ${ENV_NAME}"
echo ""

# Pinned versions shared across all modes.
TORCH_VER="2.5.1"
TORCHVISION_VER="0.20.1"
TORCHAUDIO_VER="2.5.1"
NUMPY_VER="1.26.4"
TRANSFORMERS_VER="4.57.6"

# ---------------------------------------------------------------------------
# Create / activate conda env
# ---------------------------------------------------------------------------
conda create -n "${ENV_NAME}" python=3.11 -y
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

# ---------------------------------------------------------------------------
# Per-mode toolchain
# ---------------------------------------------------------------------------
case "$MODE" in
    hpc)
        echo "[setup] HPC cluster — loading modules"
        module purge
        module load GCC/13.2.0
        module load CUDA/12.4.0
        module load CMake/3.27.6
        export CC=$(which gcc)
        export CXX=$(which g++)
        export CUDA_HOME="${CUDA_ROOT:-$CUDA_HOME}"
        export CUDACXX="$CUDA_HOME/bin/nvcc"
        export CUDAHOSTCXX="$CXX"
        ;;
    desktop)
        echo "[setup] Desktop (GPU) — using system CUDA toolkit"
        export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
        export CUDACXX="${CUDA_HOME}/bin/nvcc"
        ;;
    cpu)
        echo "[setup] CPU-only install"
        ;;
esac

# ---------------------------------------------------------------------------
# Backend-specific install
# ---------------------------------------------------------------------------

install_llama_cpp_cuda () {
    echo "[setup] Building llama-cpp-python with CUDA offload …"
    export CMAKE_ARGS="-DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON"
    export FORCE_CMAKE=1
    mkdir -p modules
    if [ ! -d "modules/llama-cpp-python" ]; then
        git clone --recurse-submodules https://github.com/abetlen/llama-cpp-python.git modules/llama-cpp-python
    fi
    cd modules/llama-cpp-python
    pip install --no-cache-dir --force-reinstall .
    cd ../..
}

install_llama_cpp_cpu () {
    echo "[setup] Installing llama-cpp-python (CPU / OpenBLAS) …"
    export CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
    export FORCE_CMAKE=1
    pip install --no-cache-dir --force-reinstall llama-cpp-python
}

install_vllm () {
    echo "[setup] Installing vLLM …"
    if [ "$MODE" = "cpu" ]; then
        echo "[warn] vLLM does not officially support CPU-only execution."
        echo "       Proceeding anyway — GPU is strongly recommended."
    fi
    pip install vllm
}

case "$BACKEND" in
    vllm)
        install_vllm
        ;;
    llama-cpp)
        case "$MODE" in
            hpc|desktop) install_llama_cpp_cuda ;;
            cpu)         install_llama_cpp_cpu  ;;
        esac
        ;;
esac

# ---------------------------------------------------------------------------
# Shared Python requirements
# ---------------------------------------------------------------------------
echo "[setup] Installing shared requirements …"
pip install -r requirements.txt

# ---------------------------------------------------------------------------
# PyTorch
# ---------------------------------------------------------------------------
if [ "$MODE" = "cpu" ]; then
    pip install --upgrade \
        "torch==${TORCH_VER}" "torchvision==${TORCHVISION_VER}" "torchaudio==${TORCHAUDIO_VER}" \
        --index-url https://download.pytorch.org/whl/cpu
else
    pip install --upgrade \
        "torch==${TORCH_VER}" "torchvision==${TORCHVISION_VER}" "torchaudio==${TORCHAUDIO_VER}" \
        --index-url https://download.pytorch.org/whl/cu124
fi

# ---------------------------------------------------------------------------
# Post-install version pins
# ---------------------------------------------------------------------------
pip install "numpy==${NUMPY_VER}"
pip install "transformers==${TRANSFORMERS_VER}"

# llama-cpp env also needs SentenceTransformers for its embedder
if [ "$BACKEND" = "llama-cpp" ]; then
    pip install sentence-transformers SentencePiece
fi

# ---------------------------------------------------------------------------
# Write the chosen backend to config.json so the TUI starts with it selected
# ---------------------------------------------------------------------------
CONFIG_FILE="config.json"
if [ -f "$CONFIG_FILE" ]; then
    # Use Python to update the JSON in-place (jq may not be available on HPC)
    python3 - <<PYEOF
import json, pathlib
p = pathlib.Path("${CONFIG_FILE}")
d = json.loads(p.read_text())
d["llm_backend"] = "${BACKEND}"
p.write_text(json.dumps(d, indent=2))
print(f"[setup] config.json updated: llm_backend = ${BACKEND}")
PYEOF
fi

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  DisTraceAI setup complete                               ║"
echo "╠═══════════════════════════════════════════════════════════╣"
echo "║  Backend : ${BACKEND}                                    "
echo "║  Env     : ${ENV_NAME}                                   "
echo "║                                                           ║"
echo "║  To start:                                               ║"
echo "║    conda activate ${ENV_NAME}                            "
echo "║    python main.py                                        ║"
echo "╚═══════════════════════════════════════════════════════════╝"
