#!/bin/bash
#SBATCH --job-name=evaluateGene
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:1
#SBATCH -G 1
#SBATCH -C "A100-40GB|A100-80GB|H100|V100-16GB|V100-32GB|RTX6000|A40|L40S"
#SBATCH --mem-per-gpu 16G
#SBATCH -n 12
#SBATCH -N 1
echo "Launching Python Evaluation"
hostname
module load cuda/12
module load anaconda3
conda activate llmge-env
# Run Python script
python ./sota/Point-Transformers/train_cls.py --model_file "models/Menghao/model_xXx7iSophHtf2q97Ps2d3rOHQ5P.py"
