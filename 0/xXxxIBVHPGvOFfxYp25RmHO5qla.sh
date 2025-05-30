#!/bin/bash
#SBATCH --job-name=llm_oper
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:1
#SBATCH -G 1
#SBATCH -C "A100-40GB|A100-80GB|H100|V100-16GB|V100-32GB|RTX6000|A40|L40S"
#SBATCH --mem-per-gpu 16G
#SBATCH -n 12
#SBATCH -N 1
echo "Launching AIsurBL"
hostname
# Load GCC version 9.2.0
# module load gcc/13.2.0
# module load cuda/11.8
module load cuda
module load anaconda3
# Activate Conda environment
conda activate llmge-env
# conda info
# Set the TOKENIZERS_PARALLELISM environment variable if needed
# export TOKENIZERS_PARALLELISM=false

# Run Python script
python src/llm_mutation.py /home/hice1/madewolu9/scratch/madewolu9/LLM_PointNet/LLM-Guided-PointCloud-Class/sota/Point-Transformers/models/Menghao/model.py /home/hice1/madewolu9/scratch/madewolu9/LLM_PointNet/LLM-Guided-PointCloud-Class/sota/Point-Transformers/models/Menghao/model_xXxxIBVHPGvOFfxYp25RmHO5qla.py 0/xXxxIBVHPGvOFfxYp25RmHO5qla_model.txt --top_p 0.1 --temperature 0.18 --apply_quality_control 'False' --hugging_face False
