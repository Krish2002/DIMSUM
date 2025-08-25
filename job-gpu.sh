#!/bin/bash

#SBATCH -J inference             # Job name
#SBATCH -p shared                # Partition name
#SBATCH --nodes=1                # Minimum number of nodes
#SBATCH --gres=gpu:1             # Number of GPUs per node
#SBATCH --ntasks 1               # no of tasks
#SBATCH --time=4:00:00           # hh:mm:ss

# Bind the required directories and run the training script


apptainer exec --bind /tmpdir,/work --nv /work/conteneurs/sessions-interactives/pytorch-24.02-py3-calmip-si.sif python infer.py