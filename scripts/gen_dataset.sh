#!/bin/bash
#SBATCH --account=p1605-25-3
#SBATCH -J "GenDataset"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH -o stdout.%J.out
#SBATCH -e stderr.%J.out

module load GCC/13.2.0
module load CUDA/12.4.0
eval "$(conda shell.bash hook)"
conda activate distrace
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}   # libcuda.so for FlashInfer
export VLLM_DEEP_GEMM_WARMUP=skip   # vLLM #41849: skip FP8 warmup (no deep_gemm; non-FP8 models)
export VLLM_USE_FLASHINFER_SAMPLER=0   # use PyTorch-native sampler; avoids FlashInfer nvcc/ninja JIT build failure
export DISABLE_KERNEL_MAPPING=1     # transformers 5.12 + kernels 0.15 import-time skew

# Full pipeline on MassiveSumm SK/CZ → three dataset CSVs.
# Output: knowledge/dataset/{subnarratives,narratives,campaigns}.csv
# No original article text is included in the output.

python $HOME/distrace/main.py \
  --generate-dataset \
  --detector models/xlm-multicw \
  --nar-extractor dense \
  --camp-extractor dense \
  --nar-embedder Qwen/Qwen3-Embedding-4B \
  --camp-embedder Qwen/Qwen3-Embedding-4B
