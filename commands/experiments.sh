python3 src/run.py -PE_matrix ./src/model/learned_PE_random_2.npy \
 -log_name train_all_test_all_random_time_PE2 > train_all_test_all_random_time_PE2.txt

python3 src/run.py -PE_matrix ./src/model/learned_PE_random_2.5.npy \
 -log_name train_all_test_all_random_time_PE2.5 > train_all_test_all_random_time_PE2.5.txt

#python3 src/run.py  -log_name gazeformer  -model Gazeformer > gazeformer.txt

python3 src/run.py  -log_name train_all_test_all_random_time_original  \
  -functionChoice original > train_all_test_all_random_time_original.txt

#python3 src/run.py  -log_name train_all_test_all_random_time > train_all_test_all_random_time.txt

'''python3 src/run.py -log_name train_wine_test_wine_random_time -testing_dataset_choice wine -training_dataset_choice wine \
       -CA_version 0     > train_wine_test_wine_random_time_fc.txt

python3 src/run.py -log_name train_yogurt_test_yogurt_random_time -testing_dataset_choice yogurt -training_dataset_choice yogurt \
       -CA_version 0     > train_yogurt_test_yogurt_random_time_fc.txt


python3 src/run.py -log_name train_wine_test_wine_random_time -testing_dataset_choice wine -training_dataset_choice wine \
       > train_wine_test_wine_random_time.txt

python3 src/run.py -log_name train_yogurt_test_yogurt_random_time -testing_dataset_choice yogurt -training_dataset_choice yogurt \
        > train_yogurt_test_yogurt_random_time.txt'''
