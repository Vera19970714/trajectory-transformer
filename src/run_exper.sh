for i in 1
do
   echo "Fold $i"
   CUDA_VISIBLE_DEVICES=0 python run.py -early_stop_patience 5 -log_dir TransformerMIT1003_evaluation  -data_folder_path ../dataset/MIT1003/ -enable_logging True -log_name $i -fold $i  -gpus 1 -batch_size 1  -num_epochs 10

done