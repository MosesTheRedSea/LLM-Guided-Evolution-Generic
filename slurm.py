import os
from src.cfg.constants import *

def replace_script_configuration(file_path, new_config):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    empty_line_index = -1
    for i, line in enumerate(lines):
        if line.strip() == '': 
            empty_line_index = i
            break
    new_lines = new_config.split('\n')
    new_lines = [line + '\n' if not line.endswith('\n') and line.strip() else line for line in new_lines]
    updated_content = new_lines + lines[empty_line_index:]
    with open(file_path, 'w') as f:
        f.writelines(updated_content)

if __name__ == "__main__":
    configuration_path = f'/home/hice1/madewolu9/scratch/madewolu9/LLMGE_Point_Cloud_Generic/LLM-Guided-Evolution-Generic/slurm-config/{CLUSTER}.txt'

    with open(configuration_path, 'r') as file:
        content = [item.strip() for item in file.readlines()]
        indices = [index for index, value in enumerate(content) if value == "------"]
        # run.sh sbatch configuration
        runsh_config_lines = "\n".join(content[indices[0]+1:indices[1]])
        
        replace_script_configuration("run.sh", runsh_config_lines)
        
        # mixt.sh sbatch configuration
        mixtsh_config_lines = "\n".join(content[indices[2]+1:indices[3]])
        
        replace_script_configuration("src/mixt.sh", mixtsh_config_lines)

        # llm-gpu sbtach configuration
        LLM_GPU = content[indices[4]+1:indices[5]][0]
   
        # python-bash-script sbatch configuration
        PYTHON_BASH_SCRIPT_TEMPLATE = "\n".join(content[indices[6]+1:indices[7]]) + """
echo "Launching Python Evaluation"
hostname

# Load GCC version 9.2.0
# module load gcc/13.2.0
module load cuda
module load anaconda3
# Activate Conda environment
#conda activate llm_guided_env
source .venv/bin/activate
export LD_LIBRARY_PATH=~/.conda/envs/llm_guided_env/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
# conda info

# Set the TOKENIZERS_PARALLELISM environment variable if needed
# export TOKENIZERS_PARALLELISM=false

# Run Python script
{}
"""

        # llm-bash-script sbatch configuration
        LLM_BASH_SCRIPT_TEMPLATE = "\n".join(content[indices[8]+1:indices[9]]) +  """
echo "Launching AIsurBL"
hostname

# Load GCC version 9.2.0
# module load gcc/13.2.0
# module load cuda/11.8
module load cuda
module load anaconda3
# Activate Conda environment
#conda activate llm_guided_env
source .venv/bin/activate
export LD_LIBRARY_PATH=~/.conda/envs/llm_guided_env/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
# conda info

# Set the TOKENIZERS_PARALLELISM environment variable if needed
# export TOKENIZERS_PARALLELISM=false

# Run Python script
{}
"""

    