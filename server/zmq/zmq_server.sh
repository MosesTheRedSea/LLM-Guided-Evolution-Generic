#!/bin/bash
#SBATCH --job-name=ZMQ-LLMGE01-Server
#SBATCH -t 5-00:00
#SBATCH --gres=gpu:2
#SBATCH -C "NVIDIAA10080GBPCIe|NVIDIAA100-SXM4-80GB|NVIDIAH100NVL"
#SBATCH --mem 160G
#SBATCH -c 16
#SBATCH -w ice193

echo "launching LLM Server"

hostname

module load cuda/12.2.2

# Make sure CUDA can see all GPUs
export CUDA_VISIBLE_DEVICES=0,1
export MKL_THREADING_LAYER=GNU

source /home/madewolu9/madewolu9_ICE/LLMGE01/LLM-Guided-Evolution-Generic/.venv/bin/

