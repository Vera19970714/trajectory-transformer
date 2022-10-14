import pytorch_lightning as pl
from models import *
from basemodels import *
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TGT_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 27, 28, 29, 30
class Conv_AutoencoderModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        torch.manual_seed(0)
        SRC_VOCAB_SIZE = 27+4
        TGT_VOCAB_SIZE = 27+4
        EMB_SIZE = 512
        NHEAD = 2 #todo: changed settings
        FFN_HID_DIM = 512
        NUM_ENCODER_LAYERS = 1
        NUM_DECODER_LAYERS = 1
        self.model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                         NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM).to(DEVICE).float()
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
        self.loggerS= SummaryWriter(f'./lightning_logs/{args.log_name}')
        self.total_step = 0

    def log_gradients_in_model(self, step):
        for tag, value in self.model.named_parameters():
            #print('-'*10)
            if value.grad is not None:
                #print(tag, value.grad.cpu())
                self.loggerS.add_histogram(tag + "/grad", value.grad.cpu(), step)
            #print('-' * 10)

    def training_step(self, batch, batch_idx):
        src_pos, src_img, tgt_pos, tgt_img = batch
        # src_pos(28, b), src_img(b, 28, w, h, 3), tgt_pos(max_len, b), tgt_img(b, max_len, w, h, 3)
        src_pos = src_pos.to(DEVICE)
        src_img = src_img.to(DEVICE)
        tgt_pos = tgt_pos.to(DEVICE)
        tgt_img = tgt_img.to(DEVICE)

        tgt_input = tgt_pos[:-1, :]
        tgt_img = tgt_img[:, :-1, :, :, :]
        # src: 15, b; tgt_input: 14, b; src_msk: 15, 15; tgt_msk: 13, 13; tgt_padding_msk: 2, 13; src_padding_msk: 2, 15
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_pos, tgt_input)

        # CHANGED position from one dimension to two dimensions
        # tgt_input: 11,1 to 11,2
        # src_pos: 28, 1 to 28, 2
        tgt_input_2d = torch.zeros((tgt_input.size()[0], tgt_input.size()[1], 2)).to(DEVICE).float()
        tgt_input_2d[:, :, 0] = tgt_input // 9
        tgt_input_2d[:, :, 1] = torch.remainder(tgt_input, 9)
        tgt_input_2d[0, :, 0] = 1.5
        tgt_input_2d[0, :, 1] = 4.5

        src_pos_2d = torch.zeros((src_pos.size()[0], src_pos.size()[1], 2)).to(DEVICE).float()
        src_pos_2d[:, :, 0] = src_pos // 9
        src_pos_2d[:, :, 1] = torch.remainder(src_pos, 9)
        src_pos_2d[0, :, 0] = -1
        src_pos_2d[0, :, 1] = -1

        logits = self.model(src_pos_2d.float(), tgt_input_2d.float(),  # src_pos, tgt_input,
                            src_img, tgt_img,
                            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        tgt_out = tgt_pos[1:, :]
        loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        #self.log_gradients_in_model(self.total_step)
        self.total_step += 1
        self.log('training_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'loss': loss, }

    def training_epoch_end(self, training_step_outputs):
        avg_loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
        self.log('training_loss_each_epoch', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)


    def validation_step(self, batch, batch_idx):
        src_pos, src_img, tgt_pos, tgt_img = batch
        # src_pos(28, b), src_img(b, 28, w, h, 3), tgt_pos(max_len, b), tgt_img(b, max_len, w, h, 3)
        src_pos = src_pos.to(DEVICE)
        src_img = src_img.to(DEVICE)
        tgt_pos = tgt_pos.to(DEVICE)
        tgt_img = tgt_img.to(DEVICE)

        tgt_input = tgt_pos[:-1, :]
        tgt_img = tgt_img[:, :-1, :, :, :]
        # src: 15, b; tgt_input: 14, b; src_msk: 15, 15; tgt_msk: 13, 13; tgt_padding_msk: 2, 13; src_padding_msk: 2, 15
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_pos, tgt_input)

        #CHANGED position from one dimension to two dimensions
        # tgt_input: 11,1 to 11,2
        # src_pos: 28, 1 to 28, 2
        tgt_input_2d = torch.zeros((tgt_input.size()[0], tgt_input.size()[1], 2)).to(DEVICE).float()
        tgt_input_2d[:, :, 0] = tgt_input // 9
        tgt_input_2d[:, :, 1] = torch.remainder(tgt_input, 9)
        tgt_input_2d[0, :, 0] = 1.5
        tgt_input_2d[0, :, 1] = 4.5

        src_pos_2d = torch.zeros((src_pos.size()[0], src_pos.size()[1], 2)).to(DEVICE).float()
        src_pos_2d[:, :, 0] = src_pos // 9
        src_pos_2d[:, :, 1] = torch.remainder(src_pos, 9)
        src_pos_2d[0, :, 0] = -1
        src_pos_2d[0, :, 1] = -1

        logits = self.model(src_pos_2d.float(), tgt_input_2d.float(),   #src_pos, tgt_input,
                            src_img, tgt_img,
                            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        tgt_out = tgt_pos[1:, :]
        loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        _, predicted = torch.max(logits, 2)
        print(predicted.view(1, -1))

        #print('gt', tgt_out.view(1, -1))

        self.log('validation_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'loss': loss,}

    def validation_epoch_end(self, validation_step_outputs):
        avg_loss = torch.stack([x['loss'] for x in validation_step_outputs]).mean()
        self.log('validation_loss_each_epoch', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)


    def test_step(self, batch, batch_idx):
        src_pos, src_img, tgt_pos, tgt_img = batch
        # src_pos(28, b), src_img(b, 28, w, h, 3), tgt_pos(max_len, b), tgt_img(b, max_len, w, h, 3)
        src_pos = src_pos.to(DEVICE)
        src_img = src_img.to(DEVICE)
        tgt_pos = tgt_pos.to(DEVICE)
        tgt_img = tgt_img.to(DEVICE)

        tgt_input = tgt_pos[:-1, :]
        tgt_img = tgt_img[:, :-1, :, :, :]
        # src: 15, b; tgt_input: 14, b; src_msk: 15, 15; tgt_msk: 13, 13; tgt_padding_msk: 2, 13; src_padding_msk: 2, 15
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_pos, tgt_input)

        # CHANGED position from one dimension to two dimensions
        # tgt_input: 11,1 to 11,2
        # src_pos: 28, 1 to 28, 2
        tgt_input_2d = torch.zeros((tgt_input.size()[0], tgt_input.size()[1], 2)).to(DEVICE).float()
        tgt_input_2d[:, :, 0] = tgt_input // 9
        tgt_input_2d[:, :, 1] = torch.remainder(tgt_input, 9)
        tgt_input_2d[0, :, 0] = 1.5
        tgt_input_2d[0, :, 1] = 4.5

        src_pos_2d = torch.zeros((src_pos.size()[0], src_pos.size()[1], 2)).to(DEVICE).float()
        src_pos_2d[:, :, 0] = src_pos // 9
        src_pos_2d[:, :, 1] = torch.remainder(src_pos, 9)
        src_pos_2d[0, :, 0] = -1
        src_pos_2d[0, :, 1] = -1

        logits = self.model(src_pos_2d.float(), tgt_input_2d.float(),  # src_pos, tgt_input,
                            src_img, tgt_img,
                            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        tgt_out = tgt_pos[1:, :]
        loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        _, predicted = torch.max(logits, 2)
        print(predicted.view(1, -1))
        self.log('testing_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'loss': loss, }

    def test_epoch_end(self, test_step_outputs):
        avg_loss = torch.stack([x['loss'].cpu().detach() for x in test_step_outputs]).mean()
        #todo: output sequence
        self.log('test_loss_each_epoch', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self): #TODO, this function is not used atm
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.scheduler_lambda1, gamma=self.args.scheduler_lambda2)
        return [optimizer], [scheduler]


class BaseModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        torch.manual_seed(0)
        VOCAB_SIZE = 27+4
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


    def test_step(self, batch, batch_idx):
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

        self.log('testing_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'loss': loss, }

    def test_epoch_end(self, test_step_outputs):
        avg_loss = torch.stack([x['loss'].cpu().detach() for x in test_step_outputs]).mean()
        #todo: output sequence
        self.log('test_loss_each_epoch', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self): #TODO, this function is not used atm
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.scheduler_lambda1, gamma=self.args.scheduler_lambda2)
        return [optimizer], [scheduler]