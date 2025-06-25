# Slurm Script Setup Guide

- This guide will demonstrate 

## run.sh

**!/bin/bash**
- shebang line in a Slurm script tells the system to use the Bash shell to execute the commands within the script. 


**#SBATCH --job-name=llm_opt**

- the **--job-name** flag allows you to specify the name of the slurm job

**#SBATCH --ntasks=2**
-

**#SBATCH --cpus-per-task=32**

**#SBATCH -c 16**

**#SBATCH --mem=160G**         

**#SBATCH --gres=gpu:2**

**#SBATCH -t 7-00:00**   

**#SBATCH -C "NVIDIAA10080GBPCIe"**


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