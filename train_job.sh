#!/bin/bash
#SBATCH --time=0-00:10:00      # Set a time limit
#SBATCH --account=def-jyzhao   # Specify the account under which this job runs
#SBATCH --mem=16000M           # Request 32GB of RAM
#SBATCH --gpus-per-node=1      # Request 1 GPU per node
#SBATCH --cpus-per-task=10     # Request 10 CPU cores per task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK     # Set the number of OpenMP threads to match the allocated CPUs.


# =====================================================
# Batch Script for Compute Canada
# Author:       Your Name
# Date:         2025-02-06
# Description:  This script runs a Python program using
#               a virtual environment and submits a 
#               SLURM job (run train.py).
#
# Usage:        $ sbatch train_job.sh
#
# Contributors: 
# - Zechen Zhou     zzhou186@uottawa.ca
# - Shun Hei Yiu    syiu017@uottawa.ca
# =====================================================

echo "Hello World"

# Provide information and management capabilities for NVIDIA GPUs
nvidia-smi

# Load needed python and cuda modules
module load python/3.12.4 cuda cudnn

# Activate your environment
source ~/prj_smp_workspace/env/bin/activate

# Upgrade pip
# pip install --no-index -upgrade pip

# Install packages
# pip install --no-index -r requirement.txt

# Variables for readability
# logdir=/home/zzhou186/scratch/saved
# datadir=/home/zzhou186/scratch/data
# datadir=/prj_smp_workspace/training_data

tensorboard --logdir=${logdir}/lightning_logs --host 0.0.0.0 --load_fast false & \
    python ~/prj_smp_workspace/train.py \
    --epochs 10
    # --model Conv \
    # --batch_size 32 \