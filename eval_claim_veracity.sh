#!/bin/bash
#SBATCH --account=p1605-25-3
#SBATCH -J "EvalVeracity"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=12:00:00
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

# Veracity evaluation benchmark.
# Paraphrases are cached to knowledge/veracity/multiclaim_test_paraphrases.json
# on first run (Gemma4-12b) and reused in subsequent runs.
# Evidence sources: multiclaim,wikipedia,web (disable any by removing from list).

python $HOME/distrace/main.py \
  --eval claim-veracity \
  --ver-sources multiclaim,wikipedia,web \
  --ver-generator gemma4-e2b \
  --ver-paraphrase-generator gemma4-12b \
  --ver-n-paraphrases 3 \
  --ver-max-turns 6 \
  --ver-token-budget 4096
