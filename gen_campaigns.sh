#!/bin/bash
#SBATCH --account=p1605-25-3
#SBATCH -J "GenCampaigns"
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

# Campaign extraction: narratives → campaigns with coordination scoring.
# Run Verify hierarchy first to populate veracity before this step.

DISTRACE=$HOME/distrace

# Verify hierarchy (FakeCTI central claims only)
python $DISTRACE/main.py \
  --generate claim-veracity \
  --ver-generator gemma4-e2b \
  --ver-precision awq4 \
  --ver-sources multiclaim,wikipedia,web

# Campaign extraction
python $DISTRACE/main.py \
  --generate campaigns \
  --camp-detector both \
  --camp-extractor dense \
  --camp-embedder Qwen/Qwen3-Embedding-4B \
  --camp-generator qwen3.5-2b \
  --camp-precision awq4 \
  --camp-assign-threshold 0.50 \
  --camp-min-new-size 2 \
  --camp-new-threshold 0.70 \
  --camp-coordination-threshold 0.40 \
  --camp-veracity-threshold 0.45
