# trajectory-transformer

How to calculate lenght, behavior, correct and heatmap?

Step1: save gaze form random benckmark, similaritysaliency benckmark and main model(select -write_output=True) file

Here is the current run.sh setting:
run.sh: srun python ./src/run.py \
        -train_datapath=./dataset/processdata/dataset_Q23_mousedel_time_train \
        -valid_datapath=./dataset/processdata/dataset_Q23_mousedel_time_val \
        -test_datapath=./dataset/processdata/dataset_Q23_mousedel_time_val \
        -checkpoint=./lightning_logs/mousedel_threedim_4_4_4_512/default/version_1/checkpoints/epoch=62-step=1637.ckpt \
        -log_name=test \
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
        -use_threedimension=True \
        -write_output=True \
        -limit_val_batches=1.0 \
        -val_check_interval=1.0 \

find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

Step2: calulate lenght, behavior, correct and heatmap from heatmapplot file, mainly check behaviorcal function, minoverlap function and heatmapplot function
(I set the max_length as 17 since under time pressure, subjects need to finish search tasks within 3 seconds, which is equivalent to 17, i.e. self generation gaze length will less than max_length threshold.)


