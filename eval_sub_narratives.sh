#!/bin/bash
#SBATCH --account=p1605-25-3
#SBATCH -J "EvalSubNarratives"
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

python $HOME/distrace/main.py \
  --eval sub-narratives \
  --subnar-detector both \
  --subnar-embedder Qwen/Qwen3-Embedding-0.6B \
  --subnar-generator qwen3.5-2b \
  --subnar-quantization Q4_K_M \
  --subnar-min-similarity 0.45 \
  --subnar-min-claims 2 \
  --subnar-hypotheticals 3
