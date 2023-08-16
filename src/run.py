import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import loggers as pl_loggers
from dataBuilders.data_builder import SearchDataModule
from dataBuilders.data_builder_base import BaseSearchDataModule
from model.transformerLightning import TransformerModel
from model.transformerLightning_load import TransformerModel_load
from benchmark.base_lightning import BaseModel
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data path and output files
    parser.add_argument('-data_path', default='./dataset/processdata/dataset_Q23_mousedel_time', type=str)
    parser.add_argument('-index_folder', default='./dataset/processdata/', type=str)
    parser.add_argument('-testing_dataset_choice', default='yogurt', type=str)  # choices: yogurt, shampoo
    parser.add_argument('-cross_dataset', default='None', type=str) # v2 choices: None, Pure, Mixed, Cross  # deprecated choices: None, Yes, No
    parser.add_argument('-package_size', type=int, default=27)
    parser.add_argument('-checkpoint', default= './lightning_logs/test/default/version_3/checkpoints/epoch=50-step=2090.ckpt', type=str)

    parser.add_argument('-log_name', default='test', type=str)
    parser.add_argument('-write_output', type=str, default='False')
    parser.add_argument('-output_path', type=str, default='./dataset/checkEvaluation/test/')
    parser.add_argument('-output_postfix', type=str, default='') # better to start with '_'
    parser.add_argument('-stochastic_iteration', type=int, default=1)

    # model settings and hyperparameters
    parser.add_argument('-model', default='Transformer', type=str) #BaseModel,
    parser.add_argument('-learning_rate', default=1e-4, type=float)
    parser.add_argument('-scheduler_lambda1', default=1, type=int)
    parser.add_argument('-scheduler_lambda2', default=0.95, type=float)
    parser.add_argument('-grad_accumulate', type=int, default=1)
    parser.add_argument('-clip_val', default=1.0, type=float)
    parser.add_argument('-limit_val_batches', default=1.0, type=float)
    parser.add_argument('-val_check_interval', default=1.0, type=float)
    parser.add_argument('-use_threedimension', type=str, default='True')

    # training settings
    parser.add_argument('-gpus', default='-1', type=str)
    parser.add_argument('-batch_size', type=int, default=20)
    parser.add_argument('-num_epochs', type=int, default=100)
    parser.add_argument('-random_seed', type=int, default=3407)
    parser.add_argument('-early_stop_patience', type=int, default=20)

    parser.add_argument('-do_train', type=str, default='False')
    parser.add_argument('-do_test', type=str, default='True')
    parser.add_argument('-do_valid', type=str, default='False')

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

    if args.model == 'Transformer':
        model = TransformerModel(args)
        search_data = SearchDataModule(args)
    if args.model == 'BaseModel':
        model = BaseModel(args)
        search_data = BaseSearchDataModule(args)
    else:
        print('Invalid model')

    model_load = TransformerModel_load(args)

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
        leanableposencoding = model.model.visual_positional_encoding.pos_embedding
        leanableposencoding = leanableposencoding[:28,0,:].cpu().detach().numpy()
        # save posencoding
        # np.save('./leanableposencoding.npy',leanableposencoding)
        # exit()
        trainer.test(model=model, dataloaders=search_data.test_loader)
    elif args.do_valid == 'True':
        model_load = model_load.load_from_checkpoint('./lightning_logs/best/default/version_2/checkpoints/epoch=45-step=1885.ckpt', args=args)
        model.model.cnn_embedding = model_load.model.cnn_embedding
        for param in model.model.cnn_embedding.parameters():
            param.requires_grad = False
        trainer.fit(model, search_data.train_loader, search_data.val_loader)
        trainer.test(model=model, dataloaders=search_data.test_loader)
