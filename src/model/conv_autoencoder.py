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
        NHEAD = 4 #todo: changed settings
        FFN_HID_DIM = 512
        NUM_ENCODER_LAYERS = 4
        NUM_DECODER_LAYERS = 4
        if self.args.use_threedimension == 'True':
            inputDim = 3
        else:
            inputDim = 2
        self.model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                         NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, inputDim, FFN_HID_DIM).to(DEVICE).float()
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

        if self.args.use_threedimension == 'True':
            src_pos_2d, tgt_input_2d, src_img, tgt_img, src_mask, tgt_mask, \
            src_padding_mask, tgt_padding_mask, src_padding_mask = self.processData3d(src_pos, src_img, tgt_pos,
                                                                                      tgt_img)
        else:
            src_pos_2d, tgt_input_2d, src_img, tgt_img, src_mask, tgt_mask, \
            src_padding_mask, tgt_padding_mask, src_padding_mask = self.processData2d(src_pos, src_img, tgt_pos,
                                                                                      tgt_img)

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

    def processData3d(self, src_pos, src_img, tgt_pos, tgt_img):
        tgt_input = tgt_pos[:-1, :]
        tgt_img = tgt_img[:, :-1, :, :, :]
        # src: 15, b; tgt_input: 14, b; src_msk: 15, 15; tgt_msk: 13, 13; tgt_padding_msk: 2, 13; src_padding_msk: 2, 15
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

        # changed to three dimension
        batch = tgt_input.size()[1]
        tgtValue = torch.tensor((0, 0, 1)).to(DEVICE).float()
        # assign the first in src_pos
        # use this for (0,0,1)
        #src_pos_2d[0, :] = tgtValue
        # use this for (x,y,1)
        src_pos_2d[0, :, 2] = 1
        for i in range(batch):
            Index = tgt_input[-1, i]
            tgt1 = torch.where(tgt_input[:, i] == Index)[0]
            tgt2 = torch.where(src_pos[:, i] == Index)[0]
            # use this for (x,y,1)
            tgt_input_2d[tgt1, i, 2] = 1
            src_pos_2d[tgt2, i, 2] = 1
            # use this for (0,0,1)
            #tgt_input_2d[tgt1, i] = tgtValue
            #src_pos_2d[tgt2, i] = tgtValue
        
        return src_pos_2d, tgt_input_2d,  src_img, tgt_img, src_mask, tgt_mask, \
               src_padding_mask, tgt_padding_mask, src_padding_mask

    def processData2d(self, src_pos, src_img, tgt_pos, tgt_img):
        tgt_input = tgt_pos[:-1, :]
        tgt_img = tgt_img[:, :-1, :, :, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_pos, tgt_input)

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

        return src_pos_2d, tgt_input_2d, src_img, tgt_img, src_mask, tgt_mask, \
               src_padding_mask, tgt_padding_mask, src_padding_mask

    def generate2DInput(self, tgt_input, src_pos):
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
        return src_pos_2d, tgt_input_2d

    def generate3DInput(self, tgt_input, src_pos):
        tgt_input_2d = torch.zeros((tgt_input.size()[0], tgt_input.size()[1], 3)).to(DEVICE).float()

        tgt_input_2d[:, :, 0] = tgt_input // 9
        tgt_input_2d[:, :, 1] = torch.remainder(tgt_input, 9)
        tgt_input_2d[0, :, 0] = 1.5
        tgt_input_2d[0, :, 1] = 4.5

        src_pos_2d = torch.zeros((src_pos.size()[0], src_pos.size()[1], 3)).to(DEVICE).float()
        src_pos_2d[:, :, 0] = src_pos // 9
        src_pos_2d[:, :, 1] = torch.remainder(src_pos, 9)

        # changed to three dimension
        batch = tgt_input.size()[1]
        tgtValue = torch.tensor((0, 0, 1)).to(DEVICE).float()
        # assign the first in src_pos
        # use this for (0,0,1)
        #src_pos_2d[0, :] = tgtValue
        # use this for (x,y,1)
        src_pos_2d[0, :, 2] = 1
        for i in range(batch):
            Index = tgt_input[-1, i]
            tgt1 = torch.where(tgt_input[:, i] == Index)[0]
            tgt2 = torch.where(src_pos[:, i] == Index)[0]
            # use this for (x,y,1)
            tgt_input_2d[tgt1, i, 2] = 1
            src_pos_2d[tgt2, i, 2] = 1
            # use this for (0,0,1)
            #tgt_input_2d[tgt1, i] = tgtValue
            #src_pos_2d[tgt2, i] = tgtValue
        return src_pos_2d, tgt_input_2d

    def validation_step(self, batch, batch_idx):
        src_pos, src_img, tgt_pos, tgt_img = batch
        # src_pos(28, b), src_img(b, 28, w, h, 3), tgt_pos(max_len, b), tgt_img(b, max_len, w, h, 3)
        src_pos = src_pos.to(DEVICE)
        src_img = src_img.to(DEVICE)
        tgt_pos = tgt_pos.to(DEVICE)
        tgt_img = tgt_img.to(DEVICE)

        if self.args.use_threedimension == 'True':
            src_pos_2d, tgt_input_2d, src_img, tgt_img, src_mask, tgt_mask, \
            src_padding_mask, tgt_padding_mask, src_padding_mask = self.processData3d(src_pos, src_img, tgt_pos, tgt_img)
        else:
            src_pos_2d, tgt_input_2d, src_img, tgt_img, src_mask, tgt_mask, \
            src_padding_mask, tgt_padding_mask, src_padding_mask = self.processData2d(src_pos, src_img, tgt_pos,
                                                                                      tgt_img)

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

    def test_max(self,src_pos, src_img, tgt_pos, tgt_img):
        tgt_input = tgt_pos[:-1, :]
        tgt_img = tgt_img[:, :-1, :, :, :]
        length = tgt_pos.size(0)
        loss = 0
        max_length = 18
        LOSS = torch.zeros((max_length, 1))-1
        GAZE = torch.zeros((max_length, 1))-1
        blank = torch.zeros((1, 4, src_img.size()[2], src_img.size()[3], 3)).to(DEVICE)
        new_src_img = torch.cat((src_img[:,1:,:,:], blank), dim=1) #31,300,186,3
        for i in range(1,length):
            if i==1:
                tgt_input = tgt_pos[:i, :]
                tgt_img_input = tgt_img[:, :i, :, :, :]
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_pos, tgt_input)
                if self.args.use_threedimension == 'True':
                    src_pos_2d, tgt_input_2d = self.generate3DInput(tgt_input, src_pos)
                else:
                    src_pos_2d, tgt_input_2d = self.generate2DInput(tgt_input, src_pos)

                logits = self.model(src_pos_2d.float(), tgt_input_2d.float(),  # src_pos, tgt_input,
                                    src_img, tgt_img_input,
                                    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
                _, predicted = torch.max(logits[-1,:,:], 1)
                tgt_out = tgt_pos[i, :]
                LOSS[i][0] = self.loss_fn(logits[-1,:,:].reshape(-1, logits[-1,:,:].shape[-1]), tgt_out.reshape(-1).long())
                GAZE[i][0] = predicted
                loss += self.loss_fn(logits[-1,:,:].reshape(-1, logits[-1,:,:].shape[-1]), tgt_out.reshape(-1).long())

                next_tgt_img_input = torch.cat((tgt_img_input, new_src_img[:, predicted, :, :, :]), dim=1)
                next_tgt_input = torch.cat((tgt_input, predicted.view(-1, 1)), dim=0)
            else:
                tgt_input = next_tgt_input
                tgt_img_input = next_tgt_img_input
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_pos, tgt_input)
                if self.args.use_threedimension == 'True':
                    src_pos_2d, tgt_input_2d = self.generate3DInput(tgt_input, src_pos)
                else:
                    src_pos_2d, tgt_input_2d = self.generate2DInput(tgt_input, src_pos)
                logits = self.model(src_pos_2d.float(), tgt_input_2d.float(),  # src_pos, tgt_input,
                                    src_img, tgt_img_input,
                                    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
                _, predicted = torch.max(logits[-1,:,:], 1)
                tgt_out = tgt_pos[i, :]
                LOSS[i][0] = self.loss_fn(logits[-1,:,:].reshape(-1, logits[-1,:,:].shape[-1]), tgt_out.reshape(-1).long())
                GAZE[i][0] = predicted
                loss += self.loss_fn(logits[-1,:,:].reshape(-1, logits[-1,:,:].shape[-1]), tgt_out.reshape(-1).long())
                next_tgt_img_input = torch.cat((next_tgt_img_input, new_src_img[:, predicted, :, :, :]), dim=1)
                next_tgt_input = torch.cat((next_tgt_input, predicted.view(-1,1)), dim=0)
                
        loss = loss / i
        return loss, LOSS, GAZE

    def test_expect(self,src_pos, src_img, tgt_pos, tgt_img):
        tgt_input = tgt_pos[:-1, :]
        tgt_img = tgt_img[:, :-1, :, :, :]
        length = tgt_pos.size(0)
        loss = 0
        blank = torch.zeros((1, 4, src_img.size()[2], src_img.size()[3], 3)).to(DEVICE)
        new_src_img = torch.cat((src_img[:,1:,:,:], blank), dim=1) #31,300,186,3
        iter = 100
        for n in range(iter):
            loss_per = 0
            for i in range(1,length):
                if i==1:
                    tgt_input = tgt_pos[:i, :]
                    tgt_img_input = tgt_img[:, :i, :, :, :]
                    # src: 15, b; tgt_input: 14, b; src_msk: 15, 15; tgt_msk: 13, 13; tgt_padding_msk: 2, 13; src_padding_msk: 2, 15
                    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_pos, tgt_input)
                    if self.args.use_threedimension == 'True':
                        src_pos_2d, tgt_input_2d = self.generate3DInput(tgt_input, src_pos)
                    else:
                        src_pos_2d, tgt_input_2d = self.generate2DInput(tgt_input, src_pos)
                
                    logits = self.model(src_pos_2d.float(), tgt_input_2d.float(),  # src_pos, tgt_input,
                                        src_img, tgt_img_input,
                                        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
                    logits_new = F.softmax(logits[-1,:,:].view(-1), dim=0)
                    predicted = torch.multinomial(logits_new,1,replacement=True)
                    tgt_out = tgt_pos[i, :]
                    loss_per += self.loss_fn(logits[-1,:,:].reshape(-1, logits[-1,:,:].shape[-1]), tgt_out.reshape(-1).long())
                    next_tgt_img_input = torch.cat((tgt_img_input, new_src_img[:, predicted, :, :, :]), dim=1)
                    next_tgt_input = torch.cat((tgt_input, predicted.view(-1, 1)), dim=0)
                
                else:
                    tgt_input = next_tgt_input
                    tgt_img_input = next_tgt_img_input
                    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_pos, tgt_input)
                    if self.args.use_threedimension == 'True':
                        src_pos_2d, tgt_input_2d = self.generate3DInput(tgt_input, src_pos)
                    else:
                        src_pos_2d, tgt_input_2d = self.generate2DInput(tgt_input, src_pos)
                    logits = self.model(src_pos_2d.float(), tgt_input_2d.float(),  # src_pos, tgt_input,
                                        src_img, tgt_img_input,
                                        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
                    logits_new = F.softmax(logits[-1,:,:].view(-1), dim=0)
                    predicted = torch.multinomial(logits_new,1,replacement=True)
                    tgt_out = tgt_pos[i, :]
                    loss_per += self.loss_fn(logits[-1,:,:].reshape(-1, logits[-1,:,:].shape[-1]), tgt_out.reshape(-1).long())
                    next_tgt_img_input = torch.cat((next_tgt_img_input, new_src_img[:, predicted, :, :, :]), dim=1)
                    next_tgt_input = torch.cat((next_tgt_input, predicted.view(-1,1)), dim=0)
                    
            loss += loss_per / i
        loss= loss / iter
        return loss

    def test_gt(self,src_pos, src_img, tgt_pos, tgt_img):
        tgt_input = tgt_pos[:-1, :]
        tgt_img_input = tgt_img[:, :-1, :, :, :]
       
        # src: 15, b; tgt_input: 14, b; src_msk: 15, 15; tgt_msk: 13, 13; tgt_padding_msk: 2, 13; src_padding_msk: 2, 15
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_pos, tgt_input)
        if self.args.use_threedimension == 'True':
            src_pos_2d, tgt_input_2d = self.generate3DInput(tgt_input, src_pos)
        else:
            src_pos_2d, tgt_input_2d = self.generate2DInput(tgt_input, src_pos)

        logits = self.model(src_pos_2d.float(), tgt_input_2d.float(),  # src_pos, tgt_input,
                            src_img, tgt_img_input,
                            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        tgt_out = tgt_pos[1:, :]
        loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        _, predicted = torch.max(logits, 2)
        print(predicted)
        return loss
        
    def test_step(self, batch, batch_idx):
        src_pos, src_img, tgt_pos, tgt_img = batch
        src_pos = src_pos.to(DEVICE)
        src_img = src_img.to(DEVICE)
        tgt_pos = tgt_pos.to(DEVICE)
        tgt_img = tgt_img.to(DEVICE)
        loss_max, LOSS, GAZE = self.test_max(src_pos, src_img, tgt_pos, tgt_img)
        loss_expect = self.test_expect(src_pos, src_img, tgt_pos, tgt_img)
        loss_gt = self.test_gt(src_pos, src_img, tgt_pos, tgt_img)
        self.log('testing_loss', loss_max, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        if self.args.write_output == 'True':
            return {'loss_max': loss_max, 'LOSS':LOSS, 'GAZE':GAZE}
        else:
            return {'loss_max': loss_max, 'loss_expect':loss_expect, 'loss_gt': loss_gt}

    def test_epoch_end(self, test_step_outputs):
        if self.args.write_output == 'True':
            avg_loss = torch.stack([x['loss'].cpu().detach() for x in test_step_outputs]).mean()
            all_loss,all_gaze = pd.DataFrame(),pd.DataFrame()
            for output in test_step_outputs:
                losses = list(output['LOSS'].cpu().detach().numpy())
                all_loss = pd.concat([all_loss, pd.DataFrame(losses)],axis=0)
                gazes = list(output['GAZE'].cpu().detach().numpy())
                all_gaze = pd.concat([all_gaze, pd.DataFrame(gazes)],axis=0)
            all_loss.reset_index().drop(['index'],axis=1)
            all_loss.replace(-1, np.nan, inplace=True)
            all_gaze.reset_index().drop(['index'],axis=1)
            all_gaze.replace(-1, np.nan, inplace=True)
            all_loss.to_excel('./dataset/outputdata/loss_gt_twodim.xlsx', index=False)
            all_gaze.to_excel('./dataset/outputdata/gaze_gt_twodim.xlsx', index=False)
            self.log('test_loss_each_epoch', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        else:
            avg_loss = torch.stack([x['loss_max'].cpu().detach() for x in test_step_outputs]).mean()
            self.log('test_loss_max_each_epoch', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)

            avg_loss = torch.stack([x['loss_expect'].cpu().detach() for x in test_step_outputs]).mean()
            self.log('test_loss_expect_each_epoch', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)

            avg_loss = torch.stack([x['loss_gt'].cpu().detach() for x in test_step_outputs]).mean()
            self.log('test_loss_gt_each_epoch', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self): 
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.scheduler_lambda1, gamma=self.args.scheduler_lambda2)
        return [optimizer], [scheduler]


