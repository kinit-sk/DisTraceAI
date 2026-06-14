conda create -n distrace python=3.11 -y && conda activate distrace

module purge
module load GCC/13.2.0
module load CUDA/12.4.0  # or conda install -c "nvidia/label/cuda-12.4.0" cuda-toolkit
module load CMake/3.27.6

export CC=$(which gcc)
export CXX=$(which g++)
export CUDA_HOME=$CUDA_ROOT
export CUDACXX=$CUDA_HOME/bin/nvcc
export CUDAHOSTCXX=$CXX

export CMAKE_ARGS="-DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON"
export FORCE_CMAKE=1

cd cd modules/
git clone --recurse-submodules https://github.com/abetlen/llama-cpp-python.git
cd llama-cpp-python/
pip install --no-cache-dir --force-reinstall .

cd ../..

pip install -r requirements.txt
pip install numpy==1.26.4
pip install --upgrade "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1" \
  --index-url https://download.pytorch.org/whl/cu121
#pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

