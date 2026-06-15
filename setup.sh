#!/bin/bash
# DisTraceAI dependency installer.
#
# Usage:
#   ./setup.sh hpc        # HPC cluster: module-based toolchain, CUDA llama.cpp
#   ./setup.sh desktop    # Local workstation with an NVIDIA GPU
#   ./setup.sh cpu        # CPU-only install (no CUDA)
#
# Each mode builds the same conda env ("distrace"), installs the Python
# requirements, and installs llama-cpp-python with the right backend.
set -euo pipefail

MODE="${1:-}"
case "$MODE" in
    hpc|desktop|cpu) ;;
    *)
        echo "Usage: $0 {hpc|desktop|cpu}" >&2
        exit 1
        ;;
esac

# Pinned versions shared across all modes.
TORCH_VER="2.5.1"
TORCHVISION_VER="0.20.1"
TORCHAUDIO_VER="2.5.1"
NUMPY_VER="1.26.4"
TRANSFORMERS_VER="4.57.6"

# --- conda env (shared) ----------------------------------------------------
conda create -n distrace python=3.11 -y
eval "$(conda shell.bash hook)"
conda activate distrace

# --- per-mode toolchain + llama-cpp-python backend -------------------------
install_llama_cpp_cuda () {
    # Build llama-cpp-python with CUDA offload enabled.
    export CMAKE_ARGS="-DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON"
    export FORCE_CMAKE=1
    pip install --no-cache-dir --force-reinstall llama-cpp-python
}

install_llama_cpp_cpu () {
    # CPU-only build (OpenBLAS); no CUDA toolchain required.
    export CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
    export FORCE_CMAKE=1
    pip install --no-cache-dir --force-reinstall llama-cpp-python
}

case "$MODE" in
    hpc)
        echo "[setup] HPC cluster install"
        module purge
        module load GCC/13.2.0
        module load CUDA/12.4.0
        module load CMake/3.27.6
        export CC=$(which gcc)
        export CXX=$(which g++)
        export CUDA_HOME="${CUDA_ROOT:-$CUDA_HOME}"
        export CUDACXX="$CUDA_HOME/bin/nvcc"
        export CUDAHOSTCXX="$CXX"
        install_llama_cpp_cuda
        ;;
    desktop)
        echo "[setup] Desktop (GPU) install"
        # Assumes a working CUDA toolkit + nvcc on PATH (or conda-installed).
        export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
        export CUDACXX="${CUDA_HOME}/bin/nvcc"
        install_llama_cpp_cuda
        ;;
    cpu)
        echo "[setup] CPU-only install"
        install_llama_cpp_cpu
        ;;
esac

# --- Python requirements (shared) ------------------------------------------
pip install -r requirements.txt

# --- torch (backend depends on mode) ---------------------------------------
if [ "$MODE" = "cpu" ]; then
    pip install --upgrade \
        "torch==${TORCH_VER}" "torchvision==${TORCHVISION_VER}" "torchaudio==${TORCHAUDIO_VER}" \
        --index-url https://download.pytorch.org/whl/cpu
else
    pip install --upgrade \
        "torch==${TORCH_VER}" "torchvision==${TORCHVISION_VER}" "torchaudio==${TORCHAUDIO_VER}" \
        --index-url https://download.pytorch.org/whl/cu121
fi

# --- post-install pins (shared) --------------------------------------------
pip install "numpy==${NUMPY_VER}"
pip install "transformers==${TRANSFORMERS_VER}"
pip install SentencePiece

echo "[setup] done (mode: ${MODE})"
