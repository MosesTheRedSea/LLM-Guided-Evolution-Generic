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

module load cuda
module load anaconda3
export CUDA_VISIBLE_DEVICES=0
export MKL_THREADING_LAYER=GNU 

export SERVER_HOSTNAME=$(hostname)

echo "INFO: Setting up the environment..."

source /home/madewolu9/madewolu9_ICE/LLMGE01/LLM-Guided-Evolution-Generic/.venv/bin/activate

python run_improved.py point_transformers_test
