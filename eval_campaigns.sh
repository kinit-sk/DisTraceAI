#!/bin/bash
#SBATCH --account=p1605-25-3
#SBATCH -J "EvalCampaigns"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH -o stdout.%J.out
#SBATCH -e stderr.%J.out

module load GCC/13.2.0
module load CUDA/12.4.0
eval "$(conda shell.bash hook)"
conda activate distrace

# Campaign clustering evaluation (ARI/NMI/V-measure) against FakeCTI ground truth.
# Requires: FakeCTI converter run + full pipeline through campaigns on FakeCTI.

python $HOME/distrace/main.py \
  --eval campaigns \
  --camp-extractor dense \
  --camp-detector both
