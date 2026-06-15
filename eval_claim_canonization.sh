#!/bin/bash
#SBATCH --account=p1605-25-3
#SBATCH -J "EvalClaimCanonization"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH -o stdout.%J.out
#SBATCH -e stderr.%J.out

module load CUDA/12.4.0

eval "$(conda shell.bash hook)"
conda activate distrace

python $HOME/distrace/main.py \
  --eval claim-canonization
