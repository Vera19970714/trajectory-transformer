#!/bin/bash
#SBATCH --job-name=BEHAVE
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=./results/slurm_output/test.log
#SBATCH --error=./results/slurm_output/test_error.log
#SBATCH --partition=gpu-share

# activate conda env
source activate aging

# Your script
srun python ./src/preprocess/randsplit.py \