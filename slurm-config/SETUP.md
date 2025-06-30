# Slurm Script Setup Guide

- This guide outlines standard Slurm directives used across scripts and provides template examples in the expected order.

## run.sh

**!/bin/bash**
- shebang line in a Slurm script tells the system to use the Bash shell to execute the commands within the script. 


**#SBATCH --job-name=llm_opt**

- Sets the name of the Slurm job to llm_opt for easy identification in job queues.

**#SBATCH --ntasks=2**
- Specifies the number of tasks to run. For MPI jobs, this is the total number of processes.

**#SBATCH --cpus-per-task=32**
- Allocates 32 CPUs per task. Useful for multi-threaded tasks.

**#SBATCH -c 16**
- Alternative syntax for --cpus-per-task=16. You should standardize usage to avoid confusion.

**#SBATCH --mem=160G**         
- Requests 160 GB of memory for the job.

**#SBATCH --gres=gpu:2**
- Requests 2 GPUs for the job.

**#SBATCH -t 7-00:00**   
- Sets a time limit of 7 days for the job.

**#SBATCH -C "NVIDIAA10080GBPCIe"**
- Requests nodes with specific constraints, e.g., NVIDIA A100 80GB GPUs.

## mixt.sh

#!/bin/bash

#SBATCH --job-name=AIsur_x1

#SBATCH --ntasks=2

#SBATCH --cpus-per-task=32

#SBATCH -c 16

#SBATCH --mem=160G     

#SBATCH --gres=gpu:2

#SBATCH -time=7-00:00    

#SBATCH -C "TeslaV100S-PCIE-32GB"


## llm-gpu

TeslaV100S-PCIE-32GB


## python-bash-script

#!/bin/bash

#SBATCH --job-name=evaluateGene

#SBATCH --ntasks=2

#SBATCH --cpus-per-task=32

#SBATCH -c 16

#SBATCH --mem=160G     

#SBATCH --gres=gpu:2

#SBATCH -time=7-00:00     

#SBATCH -C "TeslaV100S-PCIE-32GB"


## llm-bash-script

#!/bin/bash

#SBATCH --job-name=llm_oper

#SBATCH -time=5-00:00  

#SBATCH --gres=gpu:2

#SBATCH -C "{}"

#SBATCH --mem=32G

#SBATCH --ntasks=2

#SBATCH --cpus-per-task=32