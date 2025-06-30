# Slurm Script Setup Guide

This guide outlines standard Slurm directives used across scripts and provides template examples in the expected order for your HPC jobs.

## General Notes

### Shebang Line

#!/bin/bash

The shebang line tells the system to use the Bash shell to execute the script.


### Job Naming

#SBATCH --job-name=example_name

Sets the name of your Slurm job for easier identification in job queues and outputs.

### Number of Tasks

#SBATCH --ntasks=2

Specifies the total number of tasks to run. For MPI jobs, this equals the number of processes.

### CPU Allocation

#SBATCH --cpus-per-task=32

Allocates CPUs per task. Use this for multi-threaded tasks.

Note: `-c` is an alternative to `--cpus-per-task`. Standardize usage in your team to avoid confusion.


### Memory Allocation

#SBATCH --mem=160G

Requests the total memory needed for the job.


### GPU Allocation

#SBATCH --gres=gpu:2

Requests the number of GPUs needed for the job.

### Time Limit

#SBATCH -t 7-00:00

Sets a maximum runtime for the job (in this example, 7 days).

### Node Constraints

#SBATCH -C "GPU_MODEL"

Specifies node constraints, such as a specific GPU model (e.g., NVIDIA A100 80GB).


## Example Scripts

### run.sh

#!/bin/bash

#SBATCH --job-name=llm_opt

#SBATCH --ntasks=2

#SBATCH --cpus-per-task=32

#SBATCH --mem=160G

#SBATCH --gres=gpu:2

#SBATCH -t 7-00:00

#SBATCH -C "NVIDIAA10080GBPCIe"

### mixt.sh

#!/bin/bash

#SBATCH --job-name=AIsur_x1

#SBATCH --ntasks=2

#SBATCH --cpus-per-task=32

#SBATCH --mem=160G

#SBATCH --gres=gpu:2

#SBATCH -t 7-00:00

#SBATCH -C "TeslaV100S-PCIE-32GB"


### python-bash-script

#!/bin/bash

#SBATCH --job-name=evaluateGene

#SBATCH --ntasks=2

#SBATCH --cpus-per-task=32

#SBATCH --mem=160G

#SBATCH --gres=gpu:2

#SBATCH -t 7-00:00

#SBATCH -C "TeslaV100S-PCIE-32GB"


### llm-bash-script

#!/bin/bash

#SBATCH --job-name=llm_oper

#SBATCH --ntasks=2

#SBATCH --cpus-per-task=32

#SBATCH --mem=32G

#SBATCH --gres=gpu:2

#SBATCH -t 5-00:00

#SBATCH -C "{}"


## Final Tips

- Review cluster-specific documentation for available GPU models and node constraints.
- Maintain consistency in syntax and directive ordering across scripts for team clarity.
- Test scripts with minimal resources before scaling to full GPU jobs.
