#!/bin/bash
# --- Slurm Job Configuration ---
#SBATCH --job-name=CMI_bigExperiment
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=experimentOutbigExperiment.log
#SBATCH --partition=gpu2
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --mem=16GB

# --- Job Execution Metadata ---
echo "===================================================="
echo "Slurm Job ID:      $SLURM_JOB_ID"
echo "Running on host:   $(hostname)"
echo "Assigned Node(s):  $SLURM_JOB_NODELIST"
echo "Start Time:        $(date)"
echo "===================================================="

# --- Environment Setup ---
export PYTHONNOUSERSITE=1
source /home1/adoyle2025/miniconda3/etc/profile.d/conda.sh

# Activated the environment specified in your documentation
# Swap back to 'ml_project' if your cluster environment differs!
CONDA_ENV_NAME="ml_project"
echo "Activating Conda Environment: $CONDA_ENV_NAME"
conda activate $CONDA_ENV_NAME

cd /home1/adoyle2025/suave101/CMI-and-CIFAR

echo "Starting Experiment..."

python3 bigExperiment.py
