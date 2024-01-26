python3 src/run.py -index_file splitlist_all_time.txt -spp 0 \
 -data_path ./dataset/processdata/dataset_Q123_mousedel_time \
 -log_name train_all_test_all_random_time_delta -testing_dataset_choice all -training_dataset_choice all -monitor validation_ss_each_epoch \
  > train_all_test_all_random_time_ss.txt

python3 src/run.py -index_file splitlist_all_time.txt -spp 0 \
 -data_path ./dataset/processdata/dataset_Q123_mousedel_time \
 -log_name train_all_test_all_random_time_loss -testing_dataset_choice all -training_dataset_choice all -monitor validation_sim_each_epoch \
  > train_all_test_all_random_time_sim.txt