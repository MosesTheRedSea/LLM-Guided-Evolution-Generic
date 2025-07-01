#!/bin/bash
#SBATCH --job-name=AIsur_x1
#SBATCH -t 8-00:00
#SBATCH --gres=gpu:3
#SBATCH -C "A100-40GB|A100-80GB|H100|V100-16GB|V100-32GB|RTX6000|A40|L40S"
#SBATCH --mem 10G
#SBATCH -c 48
echo "Launching AIsurBL"
hostname
module load gcc/13.2.0
source ~/.bashrc
source /home/hice1/madewolu9/scratch/madewolu9/LLMGE01/LLM-Guided-Evolution-Generic/.venv/bin/activate
# Set the TOKENIZERS_PARALLELISM environment variable if needed
export TOKENIZERS_PARALLELISM=false
python llm_crossover.py '/home/hice1/madewolu9/scratch/madewolu9/LLMGE01/LLM-Guided-Evolution-Generic/sota/Point-Transformers/models/Menghao/model.py' '/home/hice1/madewolu9/scratch/madewolu9/LLMGE01/LLM-Guided-Evolution-Generic/sota/Point-Transformers/models/Menghao/model_x.py' '/home/hice1/madewolu9/scratch/madewolu9/LLMGE01/LLM-Guided-Evolution-Generic/sota/Point-Transformers/models/Menghao/model_z.py'  --top_p 0.15   --temperature 0.1 --apply_quality_control 'True' --bit 8
