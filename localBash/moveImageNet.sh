#!/bin/bash
# --- Slurm Job Configuration ---
#SBATCH --job-name=moveImageNet
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=moveImageNet.log
#SBATCH --partition=med
#SBATCH --cpus-per-task=9
#SBATCH --mem=16GB
#SBATCH --mail-user=adoyle2025@my.fit.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Remove existing directories if they exist
echo "Removing existing directories..."
rm -rf /shared/scratch/train
rm -rf /shared/scratch/val

mkdir -p /shared/scratch/adoyle2025@fit.edu/train /shared/scratch/adoyle2025@fit.edu/val

echo "Starting extraction of ImageNet shards..."
# Extract train shards using 8 parallel jobs
find /shared/datasets/imagenet-1k/shards/ -name "train-*.tar" | xargs -n 1 -P 8 -I {} tar -xf {} -C /shared/scratch/adoyle2025@fit.edu/train

# Extract val shards using 8 parallel jobs
find /shared/datasets/imagenet-1k/shards/ -name "val-*.tar" | xargs -n 1 -P 8 -I {} tar -xf {} -C /shared/scratch/adoyle2025@fit.edu/val

echo "Val shards extraction completed."
cp /shared/datasets/imagenet-1k/shards/classes.txt /shared/scratch/adoyle2025@fit.edu
