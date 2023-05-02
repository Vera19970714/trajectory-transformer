import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import loggers as pl_loggers
from dataBuilders.data_builder import SearchDataModule
from dataBuilders.data_builder_base import BaseSearchDataModule
from dataBuilders.data_builder_mit1003 import MIT1003DataModule
from dataBuilders.data_builder_mit1003_vit import MIT1003DataModule_VIT
from model.transformerLightning import TransformerModel
from benchmark.base_lightning import BaseModel
from model.transformerLightningMIT1003 import TransformerModelMIT1003
from model.transformerLightningMIT1003_vit import TransformerModelMIT1003_VIT

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data path and output files
    # NOT USED IN MIT1003
    parser.add_argument('-train_datapath', default='../dataset/processdata/dataset_Q23_mousedel_time_train', type=str)
    parser.add_argument('-valid_datapath', default='../dataset/processdata/dataset_Q23_mousedel_time_val', type=str)
    parser.add_argument('-test_datapath', default='../dataset/processdata/dataset_Q23_mousedel_time_val', type=str)
    parser.add_argument('-checkpoint', default=None, type=str)

    # parameters ONLY for MIT1003
    parser.add_argument('-data_folder_path', default='../dataset/MIT1003/', type=str)
    parser.add_argument('-processed_data_name', default='processedData', type=str)
    # todo: remember to change this with processed data name:
    parser.add_argument('-grid_partition', default='4', type=int)

    # NOTE: this mode is not used currently
    parser.add_argument('-architecture_mode', default='heatmap', type=str) #choice: heatmap, scanpath, joint
    #parser.add_argument('-subject', default='emb', type=str)
    #allSubjects = ['CNG', 'ajs', 'emb', 'ems', 'ff', 'hp', 'jcw', 'jw', 'kae', 'krl', 'po', 'tmj', 'tu', 'ya', 'zb']
    parser.add_argument('-fold', default='1', type=int)  # ten fold cross validation: 1 to 10

    parser.add_argument('-enable_logging', default='True', type=str)
    parser.add_argument('-log_dir', default='TransformerMIT1003_evaluation', type=str)
    parser.add_argument('-log_name', default='fold_1', type=str)
    parser.add_argument('-write_output', type=str, default='False')
    parser.add_argument('-output_path', type=str, default='../dataset/checkEvaluation/')
    parser.add_argument('-output_postfix', type=str, default='') # better to start with '_'
    parser.add_argument('-stochastic_iteration', type=int, default=10)
    parser.add_argument('-saliency_metric', type=str, default='False')
    parser.add_argument('-isGreedyOutput', type=str, default='True')
    
    # model settings and hyperparameters
    # choices: BaseModel,TransformerMIT1003,Transformer, TransformerMIT1003_vit
    parser.add_argument('-model', default='TransformerMIT1003', type=str)
    # architecture related choices: only for TransformerMIT1003
    parser.add_argument('-feature_extractor', default='CNN', type=str) # NOT USED # choice: CNN, LP
    parser.add_argument('-decoder_input', default='plus_feature', type=str) # choice: index, plus_feature
    parser.add_argument('-global_token', default='True', type=str) # choice: False, True

    parser.add_argument('-learning_rate', default=1e-4, type=float)
    parser.add_argument('-scheduler_lambda1', default=20, type=int)
    parser.add_argument('-scheduler_lambda2', default=0.95, type=float)
    parser.add_argument('-grad_accumulate', type=int, default=1)
    parser.add_argument('-clip_val', default=1.0, type=float)
    parser.add_argument('-limit_val_batches', default=1.0, type=float)
    parser.add_argument('-val_check_interval', default=1.0, type=float)
    parser.add_argument('-use_threedimension', type=str, default='True') #NOT USED IN MIT1003

    # training settings
    parser.add_argument('-gpus', default='0', type=str)
    parser.add_argument('-batch_size', type=int, default=2)
    parser.add_argument('-num_epochs', type=int, default=1)
    parser.add_argument('-random_seed', type=int, default=3407)
    parser.add_argument('-early_stop_patience', type=int, default=5)

    parser.add_argument('-do_train', type=str, default='True')
    parser.add_argument('-do_test', type=str, default='True')

    args = parser.parse_args()

    # random seed
    seed_everything(args.random_seed)

    # set logger
    if args.enable_logging == 'True':
        logger = pl_loggers.TensorBoardLogger(f'./lightning_logs/{args.log_dir}', name=args.log_name)
        # # save checkpoint & early stopping & learning rate decay & learning rate monitor
        checkpoint_callback = ModelCheckpoint(monitor='validation_evaluation_all',
                                              save_last=True,
                                              save_top_k=1,
                                              mode='min', )
        early_stop_callback = EarlyStopping(
            monitor='validation_evaluation_all',
            min_delta=0.00,
            patience=args.early_stop_patience,
            verbose=False,
            mode='min'
        )
        lr_monitor = LearningRateMonitor(logging_interval='step')
        enable_checkpointing = True
        callbacks = [lr_monitor, checkpoint_callback, early_stop_callback]
    else:
        logger = False
        callbacks= None
        enable_checkpointing = False

    # make dataloader & model

    if args.model == 'Transformer':
        model = TransformerModel(args)
        search_data = SearchDataModule(args)
    elif args.model == 'BaseModel':
        model = BaseModel(args)
        search_data = BaseSearchDataModule(args)
    elif args.model == 'TransformerMIT1003':
        model = TransformerModelMIT1003(args)
        search_data = MIT1003DataModule(args)
    elif args.model == 'TransformerMIT1003_vit':
        model = TransformerModelMIT1003_VIT(args)
        search_data = MIT1003DataModule_VIT(args)
    else:
        print('Invalid model')
    
    if args.checkpoint == 'None':
        args.checkpoint = None
    trainer = Trainer(deterministic=True,
                      num_sanity_val_steps=10,
                      resume_from_checkpoint=args.checkpoint,
                      logger=logger,
                      gpus=args.gpus,
                      #distributed_backend='ddp',
                      #plugins=DDPPlugin(find_unused_parameters=True),
                      gradient_clip_val=args.clip_val,
                      max_epochs=args.num_epochs,
                      limit_val_batches=args.limit_val_batches,
                      val_check_interval=args.val_check_interval,
                      accumulate_grad_batches=args.grad_accumulate,
                      fast_dev_run=False,
                      enable_checkpointing=enable_checkpointing,
                      callbacks=callbacks)

    # Fit the instantiated model to the data
    if args.do_train == 'True':
        trainer.fit(model, search_data.train_loader, search_data.val_loader)
        trainer.test(model=model, dataloaders=search_data.test_loader)
    elif args.do_test == 'True':
        model = model.load_from_checkpoint(args.checkpoint, args=args)
        trainer.test(model=model, dataloaders=search_data.test_loader)


