import pytorch_lightning as pl
from .mulT import *
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import pandas as pd
import numpy as np
import copy

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TransformerModelMIT1003_MULT(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.enableLogging = args.enable_logging
        self.model = SwinTransformer(img_size=(512,384),window_size=4).to(
            DEVICE)
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.total_step = 0
    def training_step(self, batch, batch_idx):
        src_img = torch.randn(1, 3, 512,384).to(DEVICE)
        tgt = torch.randn(512,384).softmax(dim=1).to(DEVICE)
        logits = self.model(src_img,isVertical=True)
        loss = self.loss_fn(logits.reshape(512,384), tgt.reshape(512,384))
        # self.log_gradients_in_model(self.total_step)
        self.total_step += 1
        if self.enableLogging == 'True':
            self.log('training_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'loss': loss, }

    def training_epoch_end(self, training_step_outputs):
        avg_loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
        print('training_loss_each_epoch: ', avg_loss)
        if self.enableLogging == 'True':
            self.log('training_loss_each_epoch', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        src_img = torch.randn(1, 3, 512,384).to(DEVICE)
        tgt = torch.randn(512,384).softmax(dim=1).to(DEVICE)
        logits = self.model(src_img,isVertical=True)
        loss = self.loss_fn(logits.reshape(512,384), tgt.reshape(512,384))
        # self.log_gradients_in_model(self.total_step)
        if self.enableLogging == 'True':
            self.log('validation_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'loss': loss, }

    def validation_epoch_end(self, validation_step_outputs):
        avg_loss = torch.stack([x['loss'] for x in validation_step_outputs]).mean()
        print('validation_loss_each_epoch: ', avg_loss)
        if self.enableLogging == 'True':
            self.log('validation_loss_each_epoch', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        src_img = torch.randn(1, 3, 512,384).to(DEVICE)
        tgt = torch.randn(512,384).softmax(dim=1).to(DEVICE)
        logits = self.model(src_img,isVertical=True)
        loss = self.loss_fn(logits.reshape(512,384), tgt.reshape(512,384))
        # self.log_gradients_in_model(self.total_step)
        if self.enableLogging == 'True':
            self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'loss': loss, }

    def test_epoch_end(self, test_step_outputs):
        avg_loss = torch.stack([x['loss'] for x in test_step_outputs]).mean()
        print('test_loss_each_epoch: ', avg_loss)
        if self.enableLogging == 'True':
            self.log('test_loss_each_epoch', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.scheduler_lambda1,
                                                    gamma=self.args.scheduler_lambda2)
        return [optimizer], [scheduler]

