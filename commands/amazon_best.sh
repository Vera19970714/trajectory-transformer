python3 src/run.py -data_path ./dataset/processdata/dataset_amazon \
-index_file splitlist_all_amazon.txt \
-testing_dataset_choice amazon -training_dataset_choice amazon \
-PE_matrix ./src/model/amazon_learned_random_PE.npy \
-log_name amazon_best \

