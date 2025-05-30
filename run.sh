#!/bin/bash
#SBATCH --job-name=llm_opt
#SBATCH -t 8:00:00              		# Runtime in D-HH:MM
#SBATCH --mem-per-gpu 16G
#SBATCH -n 1                          # number of CPU cores
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -C "A100-40GB|A100-80GB|H100|V100-16GB|V100-32GB|RTX6000|A40|L40S"

echo "launching LLM Guided Evolution" #1
hostname #1
# module load anaconda3/2020.07 2021.11 
module load cuda/12 #1
module load cuda #1 
module load anaconda3 #1
#export CUDA_VISIBLE_DEVICES=0
export MKL_THREADING_LAYER=GNU # Had to Change 
export SERVER_HOSTNAME=$(hostname)

conda activate llmge-env  #2
conda info 

uvicorn server:app --host $SERVER_HOSTNAME --port 8000 --workers 1 & 
sleep 5

python run_improved.py first_test