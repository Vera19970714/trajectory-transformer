import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import loggers as pl_loggers
from dataBuilders.data_builder import SearchDataModule
from dataBuilders.data_builder_base import BaseSearchDataModule
from model.transformerLightning import TransformerModel
from benchmark.base_lightning import BaseModel

import os
import sys
sys.path.append('./src/')
from evaluation.evaluation import Evaluation

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data path and output files
    parser.add_argument('-data_path', default='./dataset/processdata/dataset_Q123_mousedel_time', type=str)
    parser.add_argument('-index_folder', default='./dataset/processdata/', type=str)
    parser.add_argument('-index_file', default='splitlist_all.txt', type=str)

    parser.add_argument('-testing_dataset_choice', default='yogurt', type=str)  # wine, yogurt
    parser.add_argument('-training_dataset_choice', default='pure', type=str)  # mixed, pure

    parser.add_argument('-checkpoint', default='None', type=str)
    #parser.add_argument('-posOption', default=2, type=int) # choices: 1, 2, 3, 4
    parser.add_argument('-alpha', type=int, default=0.8)
    parser.add_argument('-functionChoice', default='exp1', type=str) # choices: linear, exp1, exp2, original
    parser.add_argument('-changeX', default='True', type=str) # None, False, True
    parser.add_argument('-CA_version', default=3, type=int)  # valid values atm: 0, 3
    # 0: no cross attention, 1: add padding to input, 2: extra FC stream, 3: add pad prob in logits
    parser.add_argument('-CA_head', default=2, type=int) # the number of cross attention heads
    parser.add_argument('-CA_dk', default=512, type=int) # 512, 64, scaling factor in attention matrix

    parser.add_argument('-log_name', default='yogurt_pure', type=str)
    parser.add_argument('-write_output', type=str, default='True')
    parser.add_argument('-output_postfix', type=str, default='') # better to start with '_'
    parser.add_argument('-stochastic_iteration', type=int, default=100)

    # model settings and hyperparameters
    parser.add_argument('-model', default='Transformer', type=str) #BaseModel,
    parser.add_argument('-learning_rate', default=1e-4, type=float)
    parser.add_argument('-scheduler_lambda1', default=1, type=int)
    parser.add_argument('-scheduler_lambda2', default=1.0, type=float)
    parser.add_argument('-grad_accumulate', type=int, default=1)
    parser.add_argument('-clip_val', default=1.0, type=float)
    parser.add_argument('-limit_val_batches', default=1.0, type=float)
    parser.add_argument('-val_check_interval', default=1.0, type=float)

    # training settings
    parser.add_argument('-gpus', default='-1', type=str)
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-num_epochs', type=int, default=100)
    parser.add_argument('-random_seed', type=int, default=888)
    parser.add_argument('-early_stop_patience', type=int, default=30)

    parser.add_argument('-monitor', type=str, default='validation_metric_each_epoch') #'validation_loss_each_epoch'
    parser.add_argument('-do_train', type=str, default='True')
    parser.add_argument('-do_test', type=str, default='True')

    args = parser.parse_args()

    if args.training_dataset_choice == 'pure':
        if args.testing_dataset_choice == 'wine':
            args.package_size = 22
            args.shelf_row = 2
            args.shelf_col = 11
        elif args.testing_dataset_choice == 'yogurt':
            args.package_size = 27
            args.shelf_row = 3
            args.shelf_col = 9
    args.output_path = './dataset/checkEvaluation/' + args.log_name + '/'

    # random seed
    seed_everything(args.random_seed)

    # create new directory of saving results
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    # set logger
    logger = pl_loggers.TensorBoardLogger(f'./lightning_logs/{args.log_name}')

    # # save checkpoint & early stopping & learning rate decay & learning rate monitor
    checkpoint_callback = ModelCheckpoint(monitor=args.monitor,
                                          save_last=True,
                                          save_top_k=1,
                                          mode='min',)

    early_stop_callback = EarlyStopping(
                            monitor=args.monitor,
                            min_delta=0.00,
                            patience=args.early_stop_patience,
                            verbose=False,
                            mode='min'
                            )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # make dataloader & model

    if args.model == 'Transformer':
        model = TransformerModel(args)
        search_data = SearchDataModule(args)
    if args.model == 'BaseModel':
        model = BaseModel(args)
        search_data = BaseSearchDataModule(args)
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
                      callbacks=[lr_monitor, checkpoint_callback, early_stop_callback])

    # Fit the instantiated model to the data
    if args.do_train == 'True':
        trainer.fit(model, search_data.train_loader, search_data.val_loader)
        trainer.test(model=model, dataloaders=search_data.test_loader)
    elif args.do_test == 'True':
        model = model.load_from_checkpoint(args.checkpoint, args=args)
        trainer.test(model=model, dataloaders=search_data.test_loader)

    e = Evaluation(args.training_dataset_choice, args.testing_dataset_choice, args.output_path,
                   ITERATION=args.stochastic_iteration, TOTAL_PCK=args.package_size)
    e.evaluation()


