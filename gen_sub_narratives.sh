#!/bin/bash
#SBATCH --account=p1605-25-3
#SBATCH -J "GenSubNarratives"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH -o stdout.%J.out
#SBATCH -e stderr.%J.out

module load GCC/13.2.0
module load CUDA/12.4.0

eval "$(conda shell.bash hook)"
conda activate distrace
export VLLM_DEEP_GEMM_WARMUP=skip   # vLLM #41849: skip FP8 warmup (no deep_gemm; non-FP8 models)
export DISABLE_KERNEL_MAPPING=1     # transformers 5.12 + kernels 0.15 import-time skew

python $HOME/distrace/main.py \
  --generate sub-narratives \
  --subnar-detector both \
  --subnar-embedder Qwen/Qwen3-Embedding-0.6B \
  --subnar-generator qwen3.5-2b \
  --subnar-precision awq4 \
  --subnar-min-similarity 0.45 \
  --subnar-min-claims 2
