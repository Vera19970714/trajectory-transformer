#!/bin/bash
export OMP_NUM_THREADS=1
for i in 1 2 3 4 5 6 7 8 9 10
do
   echo "Fold $i"
   python3 run.py -early_stop_patience 5 -log_dir TransformerMIT1003_evaluation  -data_folder_path ../dataset/MIT1003/ -enable_logging True -log_name $i -fold $i  -gpus 1 -batch_size 16  -num_epochs 100 -processed_data_name processedData_vit_N4 -grid_partition 4 -model TransformerMIT1003_vit

done
