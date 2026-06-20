#!/bin/bash
#SBATCH --account=p1605-25-3
#SBATCH -J "GenNarratives"
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

# Build the narrative hierarchy from sub-narratives.
#   --nar-extractor  dense | specfi-cs | cspecfi | context-1
#   --nar-recluster-cadence N   periodic sweep every N articles
#                               (for cspecfi this ALSO rebuilds the NodeRAG graph)
#
# dense needs no LLM; specfi-cs / cspecfi / context-1 do (and the SpecFi
# variants require the patched NodeRAG — see modules/noderag/).

python $HOME/distrace/main.py \
  --generate narratives \
  --nar-detector both \
  --nar-extractor cspecfi \
  --nar-embedder Qwen/Qwen3-Embedding-0.6B \
  --nar-generator qwen3.5-2b \
  --nar-precision awq4 \
  --nar-assign-threshold 0.55 \
  --nar-min-new-size 3 \
  --nar-new-threshold 0.75 \
  --nar-recluster-cadence 50 \
  --nar-specfi-hypotheticals 4
