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

# Veracity evaluation benchmark.
# Paraphrases are cached to knowledge/veracity/multiclaim_test_paraphrases.json
# on first run (Gemma4-12b) and reused in subsequent runs.
# Evidence sources: multiclaim,wikipedia,web (disable any by removing from list).

python $HOME/distrace/main.py \
  --eval claim-veracity \
  --ver-sources multiclaim,wikipedia,web \
  --ver-generator gemma4-e2b \
  --ver-paraphrase-generator gemma4-12b \
  --ver-quantization Q4_K_M \
  --ver-n-paraphrases 3 \
  --ver-max-turns 6 \
  --ver-token-budget 4096
