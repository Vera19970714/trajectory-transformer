for i in 1
do
   echo "Fold $i"
   CUDA_VISIBLE_DEVICES=9 python run.py -early_stop_patience 5 -log_dir TransformerMIT1003_evaluation  -data_folder_path ../dataset/MIT1003/ -enable_logging True -log_name $i -fold $i  -gpus 1 -batch_size 32  -num_epochs 100 -processed_data_name processedData_vit_N4 -grid_partition 4 -model TransformerMIT1003_vit

done