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
srun python ./src/run.py \
        -train_datapath=./dataset/processdata/dataset_Q23_time_train \
        -valid_datapath=./dataset/processdata/dataset_Q23_time_val \
        -test_datapath=./dataset/processdata/dataset_Q23_time_val \
        -checkpoint=/home/jyubj/projects/trajectory-transformer/lightning_logs/main_new/default/version_7/checkpoints/epoch=62-step=1385.ckpt \
        -log_name=main_new_test \
        -model=Conv_Autoencoder \
        -gpus='-1' \
        -batch_size=1 \
        -learning_rate=1e-4 \
        -scheduler_lambda1=1 \
        -scheduler_lambda2=0.95 \
        -num_epochs=100 \
        -grad_accumulate=1 \
        -clip_val=1.0 \
        -random_seed=3047 \
        -early_stop_patience=20 \
        -do_train=False \
        -do_test=True \
        -limit_val_batches=1.0 \
        -val_check_interval=1.0 \

find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf
