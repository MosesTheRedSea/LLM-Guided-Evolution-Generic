#!/bin/bash
#SBATCH --job-name=LLMGE01_Server
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


source /home/madewolu9/madewolu9_ICE/LLMGE01/LLM-Guided-Evolution-Generic/.venv/bin/activate

export SERVER_HOSTNAME=$(hostname)

HOSTNAME_FILE="/home/madewolu9/madewolu9_ICE/LLMGE01/LLM-Guided-Evolution-Generic/hostname.log"

echo "Writing server hostname '$SERVER_HOSTNAME' to file: $HOSTNAME_FILE"
echo "$SERVER_HOSTNAME" > "$HOSTNAME_FILE"
echo "Starting LLM server on host: $SERVER_HOSTNAME"

uvicorn server:app --host $SERVER_HOSTNAME --port 8000 --workers 1
