#!/bin/bash
#SBATCH --job-name=llm_opt
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=32
#SBATCH -c 16
#SBATCH --mem=160G
#SBATCH --gres=gpu:2
#SBATCH -t 7-00:00
#SBATCH -C "NVIDIAA10080GBPCIe"

echo "launching LLM Guided Evolution" 
hostname 
# module load anaconda3/2020.07 2021.11 
module load cuda/12 #1
module load cuda #1 
module load anaconda3 #1
# export CUDA_VISIBLE_DEVICES=0
export MKL_THREADING_LAYER=GNU 
export SERVER_HOSTNAME=$(hostname)

source .venv/bin/activate

conda info 

# export LD_LIBRARY_PATH=~/.conda/envs/llmge-env/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_ PATH

# uvicorn server:app --host $SERVER_HOSTNAME --port 8002 --workers 1 & 
# sleep 5

# Set Slurm Configurations

python slurm.py

python run_improved.py first_test