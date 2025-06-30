#!/bin/bash
#SBATCH --job-name=AIsur_x1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=32
#SBATCH -c 16
#SBATCH --mem=160G
#SBATCH --gres=gpu:2
#SBATCH -time=7-00:00
#SBATCH -C "TeslaV100S-PCIE-32GB"
echo "Launching AIsurBL"
hostname
module load gcc/13.2.0
source 
# Set the TOKENIZERS_PARALLELISM environment variable if needed
export TOKENIZERS_PARALLELISM=false
python llm_crossover.py '/gv1/projects/AI_Surrogate/dev/dev/clint/CodeLLama/codellama/LLM-Guided-Evolution-Generic/sota/PT/Point-Transformers/models/Menghao/model.py' 'LLM-Guided-Evolution-Generic/sota/PT/Point-Transformers/models/Menghao/model_x.py' 'LLM-Guided-Evolution-Generic/sota/PT/Point-Transformers/models/Menghao/model_z.py'  --top_p 0.15   --temperature 0.1 --apply_quality_control 'True' --bit 8
