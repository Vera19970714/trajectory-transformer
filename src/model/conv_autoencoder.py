import pytorch_lightning as pl
from models import *
from basemodels import *
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import pandas as pd
import numpy as np


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

        src_pos_2d, tgt_input_2d, src_img, tgt_img, src_mask, tgt_mask, \
        src_padding_mask, tgt_padding_mask, src_padding_mask = self.processData(src_pos, src_img, tgt_pos, tgt_img)

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

    def processData(self, src_pos, src_img, tgt_input, tgt_img_input):
        
        # tgt_input = tgt_pos[:-1, :]
        # tgt_img = tgt_img[:, :-1, :, :, :]
        # src: 15, b; tgt_input: 14, b; src_msk: 15, 15; tgt_msk: 13, 13; tgt_padding_msk: 2, 13; src_padding_msk: 2, 15
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_pos, tgt_input)

        # CHANGED position from one dimension to two dimensions
        # tgt_input: 11,1 to 11,2
        # src_pos: 28, 1 to 28, 2
        
        tgt_input_2d = torch.zeros((tgt_input.size()[0], tgt_input.size()[1], 3)).to(DEVICE).float()
        
        tgt_input_2d[:, :, 0] = tgt_input // 9
        tgt_input_2d[:, :, 1] = torch.remainder(tgt_input, 9)
        tgt_input_2d[0, :, 0] = 1.5
        tgt_input_2d[0, :, 1] = 4.5

        src_pos_2d = torch.zeros((src_pos.size()[0], src_pos.size()[1], 3)).to(DEVICE).float()
        src_pos_2d[:, :, 0] = src_pos // 9
        src_pos_2d[:, :, 1] = torch.remainder(src_pos, 9)
        # src_pos_2d[0, :, 0] = -1
        # src_pos_2d[0, :, 1] = -1

        # changed to three dimension
        batch = tgt_input.size()[1]
        for i in range(batch):
            Index = tgt_input[-1, i]
            tgt1 = torch.where(tgt_input[:, i] == Index)[0]
            tgt2 = torch.where(src_pos[:, i] == Index)[0]
            tgt_input_2d[tgt1, i, 2] = 1
            src_pos_2d[tgt2, i, 2] = 1
        
        return src_pos_2d, tgt_input_2d,  src_img, tgt_img_input, src_mask, tgt_mask, \
               src_padding_mask, tgt_padding_mask, src_padding_mask


    def validation_step(self, batch, batch_idx):
        src_pos, src_img, tgt_pos, tgt_img = batch
        # src_pos(28, b), src_img(b, 28, w, h, 3), tgt_pos(max_len, b), tgt_img(b, max_len, w, h, 3)
        src_pos = src_pos.to(DEVICE)
        src_img = src_img.to(DEVICE)
        tgt_pos = tgt_pos.to(DEVICE)
        tgt_img = tgt_img.to(DEVICE)

        src_pos_2d, tgt_input_2d, src_img, tgt_img, src_mask, tgt_mask, \
        src_padding_mask, tgt_padding_mask, src_padding_mask = self.processData(src_pos, src_img, tgt_pos, tgt_img)

        logits = self.model(src_pos_2d.float(), tgt_input_2d.float(),   #src_pos, tgt_input,
                            src_img, tgt_img,
                            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        # logits: 11, 1, 31, tgt_out: 11, 1
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
        print(tgt_pos)
        tgt_img = tgt_img.to(DEVICE)
        # CHANGED: the first one is discarded
        src_pos = src_pos[1:]
        src_img = src_img[:, 1:]
        max_length = 18
        length = tgt_pos.size(0)
        loss = 0
        blank = torch.zeros((1, 4, src_img.size()[2], src_img.size()[3], 3)).cuda()
        new_src_img = torch.cat((src_img, blank), dim=1) #31,300,186,3
        end_token = 30*torch.ones((max_length-length,1)).cuda()
        new_tgt_pos = torch.cat((tgt_pos, end_token), dim=0)
        iter =90
        LOSS = torch.zeros((max_length, iter))-1
        GAZE= torch.zeros((max_length, iter))-1
        for n in range(iter):
            loss_per = 0
            for i in range(1,max_length):
                if i==1:
                    tgt_input = tgt_pos[:i, :]
                    tgt_img_input = tgt_img[:, :i, :, :, :]
                    src_pos_2d, tgt_input_2d, src_img, tgt_img, src_mask, tgt_mask, \
                    src_padding_mask, tgt_padding_mask, src_padding_mask = self.processData(src_pos, src_img, tgt_input,tgt_img_input)

                    logits = self.model(src_pos_2d.float(), tgt_input_2d.float(),  # src_pos, tgt_input,
                                        src_img, tgt_img,
                                        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
                    logits_new = F.softmax(logits[-1,:,:].view(-1), dim=0)
                    predicted = torch.multinomial(logits_new,1,replacement=True)
                    tgt_out = new_tgt_pos[i, :]
                    LOSS[i][n] = self.loss_fn(logits[-1,:,:].reshape(-1, logits[-1,:,:].shape[-1]), tgt_out.reshape(-1).long())
                    GAZE[i][n] = predicted
                    loss_per += self.loss_fn(logits[-1,:,:].reshape(-1, logits[-1,:,:].shape[-1]), tgt_out.reshape(-1).long())
                    if predicted==30:
                        break
                    else:
                        next_tgt_img_input = src_img[:, predicted, :, :, :]
                        next_tgt_input = predicted.view(-1,1)
                else:
                    tgt_input = next_tgt_input
                    tgt_img_input = next_tgt_img_input
                    src_pos_2d, tgt_input_2d, src_img, tgt_img, src_mask, tgt_mask, \
                    src_padding_mask, tgt_padding_mask, src_padding_mask = self.processData(src_pos, src_img, tgt_input,tgt_img_input)
                    
                    logits = self.model(src_pos_2d.float(), tgt_input_2d.float(),  # src_pos, tgt_input,
                                        src_img, tgt_img,
                                        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
                    logits_new = F.softmax(logits[-1,:,:].view(-1), dim=0)
                    
                    predicted = torch.multinomial(logits_new,1,replacement=True)
                    tgt_out = new_tgt_pos[i, :]
                    LOSS[i][n] = self.loss_fn(logits[-1,:,:].reshape(-1, logits[-1,:,:].shape[-1]), tgt_out.reshape(-1).long())
                    GAZE[i][n] = predicted
                    loss_per += self.loss_fn(logits[-1,:,:].reshape(-1, logits[-1,:,:].shape[-1]), tgt_out.reshape(-1).long())
                    if predicted==30:
                        break
                    else:
                        # print(predicted)
                        next_tgt_img_input = torch.cat((next_tgt_img_input, new_src_img[:, predicted, :, :, :]), dim=1)
                        next_tgt_input = torch.cat((next_tgt_input, predicted.view(-1,1)), dim=0)
            loss += loss_per / i
        loss= loss / iter
        print(loss)
        self.log('testing_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'loss': loss,'gaze':GAZE,'LOSS':LOSS}

    def test_epoch_end(self, test_step_outputs):
        avg_loss = torch.stack([x['loss'].cpu().detach() for x in test_step_outputs]).mean()
        #todo: output sequence
        all_gazes, all_loss = pd.DataFrame(), pd.DataFrame()
        for output in test_step_outputs:
            gazes = list(output['gaze'].cpu().detach().numpy()) # predicted values
            losses = list(output['LOSS'].cpu().detach().numpy())
            all_loss = pd.concat([all_loss, pd.DataFrame(losses)],axis=0)
            all_gazes = pd.concat([all_gazes, pd.DataFrame(gazes)],axis=0)
        all_loss.reset_index().drop(['index'],axis=1)
        all_loss.replace(-1, np.nan, inplace=True)
        all_loss.to_excel('./dataset/outputdata/loss_twodim.xlsx', index=False)

        all_gazes.reset_index().drop(['index'],axis=1)
        all_gazes.replace(-1, np.nan, inplace=True)
        all_gazes.to_excel('./dataset/outputdata/gaze_twodim.xlsx', index=False)
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