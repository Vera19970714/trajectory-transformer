python3 src/run.py -index_file splitlist_all_no.txt -spp 3 \
 -data_path ./dataset/processdata/dataset_Q123_mousedel_no_raw \
 -log_name train_all_test_all_random_no_spp -testing_dataset_choice all -training_dataset_choice all \
  > train_all_test_all_random_no_spp.txt

python3 src/run.py -index_file splitlist_all_no.txt -spp 3 \
 -data_path ./dataset/processdata/dataset_Q123_mousedel_no_raw \
 -log_name train_wine_test_wine_random_no_spp -testing_dataset_choice wine -training_dataset_choice wine \
  > train_wine_test_wine_random_no_spp.txt

python3 src/run.py -index_file splitlist_all_no.txt -spp 3 \
 -data_path ./dataset/processdata/dataset_Q123_mousedel_no_raw \
 -log_name train_yogurt_test_yogurt_random_no_spp -testing_dataset_choice yogurt -training_dataset_choice yogurt \
  > train_yogurt_test_yogurt_random_no_spp.txt

python3 src/run.py -index_file splitlist_all_no.txt -spp 3 \
 -data_path ./dataset/processdata/dataset_Q123_mousedel_no_raw \
 -log_name train_yogurt_test_wine_random_no_spp -testing_dataset_choice wine -training_dataset_choice yogurt \
  > train_yogurt_test_wine_random_no_spp.txt

python3 src/run.py -index_file splitlist_all_no.txt -spp 3 \
 -data_path ./dataset/processdata/dataset_Q123_mousedel_no_raw \
 -log_name train_wine_test_yogurt_random_no_spp -testing_dataset_choice yogurt -training_dataset_choice wine \
  > train_wine_test_yogurt_random_no_spp.txt

python3 src/run.py -index_file splitlist_all_no.txt -spp 3 \
 -data_path ./dataset/processdata/dataset_Q123_mousedel_no_raw \
 -log_name train_all_test_wine_random_no_spp -testing_dataset_choice wine -training_dataset_choice all \
  > train_all_test_wine_random_no_spp.txt

python3 src/run.py -index_file splitlist_all_no.txt -spp 3 \
 -data_path ./dataset/processdata/dataset_Q123_mousedel_no_raw \
 -log_name train_all_test_yogurt_random_no_spp -testing_dataset_choice yogurt -training_dataset_choice all \
  > train_all_test_yogurt_random_no_spp.txt

python3 src/run.py -index_file splitlist_all_no.txt -spp 3 \
 -data_path ./dataset/processdata/dataset_Q123_mousedel_no_raw \
 -log_name train_wine_test_all_random_no_spp -testing_dataset_choice all -training_dataset_choice wine \
  > train_wine_test_all_random_no_spp.txt

python3 src/run.py -index_file splitlist_all_no.txt -spp 3 \
 -data_path ./dataset/processdata/dataset_Q123_mousedel_no_raw \
 -log_name train_yogurt_test_all_random_no_spp -testing_dataset_choice all -training_dataset_choice yogurt \
  > train_yogurt_test_all_random_no_spp.txt
