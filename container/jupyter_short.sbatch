#!/bin/bash
#SBATCH --job-name=thesis-jupyter
#SBATCH --partition=me-cor,general
#SBATCH --qos=short
#SBATCH --account=me-cor
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:a40:1
#SBATCH --output=/tudelft.net/staff-umbrella/TeamHolgerResearch/lleba/Thesis/container/jupyter_%j.out
#SBATCH --error=/tudelft.net/staff-umbrella/TeamHolgerResearch/lleba/Thesis/container/jupyter_%j.err

set -e

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

srun /tudelft.net/staff-umbrella/TeamHolgerResearch/lleba/Thesis/container/start_jupyter.sh
