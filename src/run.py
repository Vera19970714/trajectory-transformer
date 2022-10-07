import argparse
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.plugins import DDPPlugin
from data_builder import SearchDataModule, BaseSearchDataModule
from model.conv_autoencoder import Conv_AutoencoderModel, BaseModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data path
    parser.add_argument('-train_datapath', default='../dataset/processdata/dataset_Q23_time_train', type=str)
    parser.add_argument('-valid_datapath', default='../dataset/processdata/dataset_Q23_time_val', type=str)
    parser.add_argument('-test_datapath', default='../dataset/processdata/dataset_Q23_time_val', type=str)
    parser.add_argument('-checkpoint', default='../ckpt/epoch=17-step=395.ckpt', type=str)
    parser.add_argument('-log_name', default='test_log', type=str)
    # model setting
    parser.add_argument('-model', default='Conv_Autoencoder', type=str)
    # training hyperparameters
    parser.add_argument('-gpus', default='0', type=str)
    parser.add_argument('-batch_size', type=int, default=2)
    parser.add_argument('-learning_rate', default=3e-5, type=float)
    parser.add_argument('-scheduler_lambda1', default=20, type=int)
    parser.add_argument('-scheduler_lambda2', default=0.95, type=float)
    parser.add_argument('-num_epochs', type=int, default=100)
    parser.add_argument('-grad_accumulate', type=int, default=1)
    parser.add_argument('-clip_val', default=1.0, type=float)
    parser.add_argument('-random_seed', type=int, default=3407)
    parser.add_argument('-early_stop_patience', type=int, default=5)
    parser.add_argument('-do_train', type=str, default='True')
    parser.add_argument('-do_test', type=str, default='True')
    parser.add_argument('-limit_val_batches', default=1.0, type=float)
    parser.add_argument('-val_check_interval', default=1.0, type=float)

    args = parser.parse_args()

    # random seed
    seed_everything(args.random_seed)

    # set logger
    logger = pl_loggers.TensorBoardLogger(f'./lightning_logs/{args.log_name}')

    # # save checkpoint & early stopping & learning rate decay & learning rate monitor
    checkpoint_callback = ModelCheckpoint(monitor='validation_loss_each_epoch',
                                          save_last=True,
                                          save_top_k=1,
                                          mode='min',)

    early_stop_callback = EarlyStopping(
                            monitor='validation_loss_each_epoch',
                            min_delta=0.00,
                            patience=args.early_stop_patience,
                            verbose=False,
                            mode='min'
                            )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # make dataloader & model
    # search_data = SearchDataModule(args)
    search_data = BaseSearchDataModule(args)
    if args.model == 'Conv_Autoencoder':
        model = Conv_AutoencoderModel(args)
    if args.model == 'BaseModel':
        model = BaseModel(args)
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
                      gradient_clip_val=1.0,
                      max_epochs=args.num_epochs,
                      limit_val_batches=args.limit_val_batches,
                      val_check_interval=args.val_check_interval,
                      accumulate_grad_batches=args.grad_accumulate,
                      fast_dev_run=False,
                      callbacks=[lr_monitor, checkpoint_callback, early_stop_callback])

    # Fit the instantiated model to the data
    if args.do_train == 'True':
        trainer.fit(model, search_data.train_loader, search_data.val_loader)
        trainer.test(model=model, dataloaders=search_data.test_loader)
    if args.do_test == 'True':
        model = model.load_from_checkpoint(args.checkpoint, args=args)
        trainer.test(model=model, dataloaders=search_data.test_loader)


