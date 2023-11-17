python3 src/run.py -PE_matrix ./src/model/learned_PE_random_2.npy \
 -log_name train_all_test_all_random_time_PE2 > train_all_test_all_random_time_PE2.txt

python3 src/run.py -PE_matrix ./src/model/learned_PE_random_2.5.npy \
 -log_name train_all_test_all_random_time_PE2.5 > train_all_test_all_random_time_PE2.5.txt

#python3 src/run.py  -log_name gazeformer  -model Gazeformer > gazeformer.txt

python3 src/run.py  -log_name train_all_test_all_random_time_original  \
  -functionChoice original > train_all_test_all_random_time_original.txt

#python3 src/run.py  -log_name train_all_test_all_random_time > train_all_test_all_random_time.txt

python3 src/run.py -log_name train_wine_test_wine_random_time_fc -testing_dataset_choice wine -training_dataset_choice wine \
       -CA_version 0     > train_wine_test_wine_random_time_fc.txt

python3 src/run.py -log_name train_yogurt_test_yogurt_random_time_fc -testing_dataset_choice yogurt -training_dataset_choice yogurt \
       -CA_version 0     > train_yogurt_test_yogurt_random_time_fc.txt


python3 src/run.py -log_name train_wine_test_wine_random_time -testing_dataset_choice wine -training_dataset_choice wine \
       > train_wine_test_wine_random_time.txt

python3 src/run.py -log_name train_yogurt_test_yogurt_random_time -testing_dataset_choice yogurt -training_dataset_choice yogurt \
        > train_yogurt_test_yogurt_random_time.txt


python3 src/run.py  -log_name train_all_test_all_random_time_comb -leave_one_comb_out 1 > train_all_test_all_random_time_comb.txt


python3 src/run.py  -log_name train_all_test_all_random_time_comb1 -leave_one_comb_out 1 \
-leave_one_comb_out_tgt_id T1_4 -leave_one_comb_out_layout_id Q1_70 \
 > train_all_test_all_random_time_comb1.txt

python3 src/run.py  -log_name train_all_test_all_random_time_comb2 -leave_one_comb_out 1 \
-leave_one_comb_out_tgt_id T3_12 -leave_one_comb_out_layout_id Q3_10 \
 > train_all_test_all_random_time_comb2.txt

 python3 src/run.py  -log_name train_all_test_all_random_time_comb3 -leave_one_comb_out 1 \
-leave_one_comb_out_tgt_id T3_15 -leave_one_comb_out_layout_id Q3_12 \
 > train_all_test_all_random_time_comb3.txt

  python3 src/run.py  -log_name train_all_test_all_random_time_comb4 -leave_one_comb_out 1 \
-leave_one_comb_out_tgt_id T1_16 -leave_one_comb_out_layout_id Q1_99 \
 > train_all_test_all_random_time_comb4.txt