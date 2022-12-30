import pytorch_lightning as pl
from models import *
from basemodels import *
import torch
import torch.nn as nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TGT_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 27, 28, 29, 30

class BaseModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        torch.manual_seed(0)
        VOCAB_SIZE = 27 + 4
        EMB_SIZE = 256
        CNN_SIZE = 1280
        HIDDEN_SIZE = 512
        NUM_DECODER_LAYERS = 3
        self.model = DecoderRNN(EMB_SIZE, HIDDEN_SIZE, CNN_SIZE, VOCAB_SIZE,
                                NUM_DECODER_LAYERS).to(DEVICE)
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    def training_step(self, batch, batch_idx):
        src_pos, question_img, src_img, tgt_pos, tgt_img = batch
        # src_pos(1, b), src_img(b, 28, w, h, 3), tgt_pos(max_len, b), tgt_img(b, max_len, w, h, 3)
        src_pos = src_pos.to(DEVICE)
        question_img = question_img.to(DEVICE)
        src_img = src_img.to(DEVICE)
        tgt_pos = tgt_pos.to(DEVICE)
        tgt_img = tgt_img.to(DEVICE)

        tgt_input = tgt_pos[:-1, :]
        tgt_img = tgt_img[:, :-1, :, :, :]

        logits = self.model(tgt_input, src_img,
                            question_img, tgt_img)
        tgt_out = tgt_pos[:, :]
        loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        self.log('training_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'loss': loss, }

    def training_epoch_end(self, training_step_outputs):
        avg_loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
        self.log('training_loss_each_epoch', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        src_pos, question_img, src_img, tgt_pos, tgt_img = batch
        # src_pos(1, b), src_img(b, 28, w, h, 3), tgt_pos(max_len, b), tgt_img(b, max_len, w, h, 3)
        src_pos = src_pos.to(DEVICE)
        question_img = question_img.to(DEVICE)
        src_img = src_img.to(DEVICE)
        tgt_pos = tgt_pos.to(DEVICE)
        tgt_img = tgt_img.to(DEVICE)

        tgt_input = tgt_pos[:-1, :]
        tgt_img = tgt_img[:, :-1, :, :, :]

        logits = self.model(tgt_input, src_img,
                            question_img, tgt_img)
        tgt_out = tgt_pos[:, :]
        loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        _, predicted = torch.max(logits, 2)
        print(predicted)

        self.log('validation_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'loss': loss, }

    def validation_epoch_end(self, validation_step_outputs):
        avg_loss = torch.stack([x['loss'] for x in validation_step_outputs]).mean()
        self.log('validation_loss_each_epoch', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_gt(self, src_pos, question_img, src_img, tgt_pos, tgt_img):
        tgt_input = tgt_pos[:-1, :]
        tgt_img = tgt_img[:, :-1, :, :, :]

        logits = self.model(tgt_input, src_img,
                            question_img, tgt_img)
        tgt_out = tgt_pos[:, :]
        loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        _, predicted = torch.max(logits, 2)
        print(predicted)
        return loss

    def test_max(self, src_pos, question_img, src_img, tgt_pos, tgt_img):
        tgt_input = tgt_pos[:-1, :]
        tgt_img = tgt_img[:, :-1, :, :, :]
        length = tgt_pos.size(0)
        loss = 0
        for i in range(1, length):
            if i == 1:
                tgt_input = tgt_pos[:i, :]
                tgt_img_input = tgt_img[:, :i, :, :, :]
                # src: 15, b; tgt_input: 14, b; src_msk: 15, 15; tgt_msk: 13, 13; tgt_padding_msk: 2, 13; src_padding_msk: 2, 15
                logits = self.model(tgt_input, src_img,
                                    question_img, tgt_img_input)
                tgt_out = tgt_pos[:2, :]
                _, predicted = torch.max(logits[:, -1, :], 1)
                print(predicted)
                loss1 = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1).long())
                next_tgt_input = torch.cat((tgt_input, predicted.view(-1, 1)), dim=0)
            else:
                # tgt_input = next_tgt_input
                tgt_input = tgt_pos[:i, :]
                tgt_img_input = tgt_img[:, :i, :, :, :]
                logits = self.model(tgt_input, src_img,
                                    question_img, tgt_img_input)
                tgt_out = tgt_pos[i, :]
                _, predicted = torch.max(logits[:, -1, :], 1)
                print(predicted)
                loss += self.loss_fn(logits[:, -1, :].reshape(-1, logits[:, -1, :].shape[-1]),
                                     tgt_out.reshape(-1).long())
                next_tgt_input = torch.cat((next_tgt_input, predicted.view(-1, 1)), dim=0)
        loss = (2 * loss1 + loss) / (i + 1)
        return loss

    def test_step(self, batch, batch_idx):
        src_pos, question_img, src_img, tgt_pos, tgt_img = batch
        # src_pos(1, b), src_img(b, 28, w, h, 3), tgt_pos(max_len, b), tgt_img(b, max_len, w, h, 3)
        src_pos = src_pos.to(DEVICE)
        question_img = question_img.to(DEVICE)
        src_img = src_img.to(DEVICE)
        tgt_pos = tgt_pos.to(DEVICE)
        tgt_img = tgt_img.to(DEVICE)
        # loss = self.test_gt(src_pos, question_img, src_img, tgt_pos, tgt_img)
        loss = self.test_max(src_pos, question_img, src_img, tgt_pos, tgt_img)
        self.log('testing_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'loss': loss, }

    def test_epoch_end(self, test_step_outputs):
        avg_loss = torch.stack([x['loss'].cpu().detach() for x in test_step_outputs]).mean()
        # todo: output sequence
        self.log('test_loss_each_epoch', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):  # TODO, this function is not used atm
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.scheduler_lambda1,
                                                    gamma=self.args.scheduler_lambda2)
        return [optimizer], [scheduler]