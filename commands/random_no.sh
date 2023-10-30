python3 src/run.py -index_file splitlist_all_no.txt -spp 0 \
 -data_path ./dataset/processdata/dataset_Q123_mousedel_no \
 -log_name train_all_test_all_random_no -testing_dataset_choice all -training_dataset_choice all \
  > train_all_test_all_random_no.txt

python3 src/run.py -index_file splitlist_all_no.txt -spp 0 \
 -data_path ./dataset/processdata/dataset_Q123_mousedel_no \
 -log_name train_wine_test_wine_random_no -testing_dataset_choice wine -training_dataset_choice wine \
  > train_wine_test_wine_random_no.txt

python3 src/run.py -index_file splitlist_all_no.txt -spp 0 \
 -data_path ./dataset/processdata/dataset_Q123_mousedel_no \
 -log_name train_yogurt_test_yogurt_random_no -testing_dataset_choice yogurt -training_dataset_choice yogurt \
  > train_yogurt_test_yogurt_random_no.txt

python3 src/run.py -index_file splitlist_all_no.txt -spp 0 \
 -data_path ./dataset/processdata/dataset_Q123_mousedel_no \
 -log_name train_yogurt_test_wine_random_no -testing_dataset_choice wine -training_dataset_choice yogurt \
  > train_yogurt_test_wine_random_no.txt

python3 src/run.py -index_file splitlist_all_no.txt -spp 0 \
 -data_path ./dataset/processdata/dataset_Q123_mousedel_no \
 -log_name train_wine_test_yogurt_random_n -testing_dataset_choice yogurt -training_dataset_choice wine \
  > train_wine_test_yogurt_random_no.txt

python3 src/run.py -index_file splitlist_all_no.txt -spp 0 \
 -data_path ./dataset/processdata/dataset_Q123_mousedel_no\
 -log_name train_all_test_wine_random_no -testing_dataset_choice wine -training_dataset_choice all \
  > train_all_test_wine_random_no.txt

python3 src/run.py -index_file splitlist_all_no.txt -spp 0 \
 -data_path ./dataset/processdata/dataset_Q123_mousedel_no \
 -log_name train_all_test_yogurt_random_n -testing_dataset_choice yogurt -training_dataset_choice all \
  > train_all_test_yogurt_random_no.txt

python3 src/run.py -index_file splitlist_all_no.txt -spp 0 \
 -data_path ./dataset/processdata/dataset_Q123_mousedel_no \
 -log_name train_wine_test_all_random_no -testing_dataset_choice all -training_dataset_choice wine \
  > train_wine_test_all_random_no.txt

python3 src/run.py -index_file splitlist_all_no.txt -spp 0 \
 -data_path ./dataset/processdata/dataset_Q123_mousedel_no \
 -log_name train_yogurt_test_all_random_no -testing_dataset_choice all -training_dataset_choice yogurt \
  > train_yogurt_test_all_random_no.txt

