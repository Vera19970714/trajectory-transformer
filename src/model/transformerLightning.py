import pytorch_lightning as pl
from .models import *
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import pandas as pd
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TransformerModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        args.package_size = 27
        self.args = args
        torch.manual_seed(0)
        SRC_VOCAB_SIZE = self.args.package_size+4
        TGT_VOCAB_SIZE = self.args.package_size+4
        self.TGT_IDX = self.args.package_size
        self.PAD_IDX = self.args.package_size+1
        self.BOS_IDX = self.args.package_size+2
        self.EOS_IDX = self.args.package_size+3
        EMB_SIZE = 512
        NHEAD = 4
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
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.PAD_IDX)
        self.norm = torch.nn.Softmax(dim=1)
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, betas=(0.9, 0.98), eps=1e-9)
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
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_pos, tgt_input, self.PAD_IDX)

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
        #tgtValue = torch.tensor((0, 0, 1)).to(DEVICE).float()
        # assign the first in src_pos
        # use this for (0,0,1)
        #src_pos_2d[0, :] = tgtValue
        # use this for (x,y,1)
        src_pos_2d[0, :, 2] = 1
        for i in range(batch):
            #Index = tgt_input[-1, i]
            Index = src_pos[0, i]
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
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_pos, tgt_input, self.PAD_IDX)

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
        #tgtValue = torch.tensor((0, 0, 1)).to(DEVICE).float()
        # assign the first in src_pos
        # use this for (0,0,1)
        #src_pos_2d[0, :] = tgtValue
        # use this for (x,y,1)
        src_pos_2d[0, :, 2] = 1
        for i in range(batch):
            #Index = tgt_input[-1, i]
            Index = src_pos[0, i]
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
        max_length = 16
        LOSS = torch.zeros((length-1, 1))-1
        GAZE = torch.zeros((max_length, 1))-1
        LOGITS = torch.zeros((max_length, 31))
        blank = torch.zeros((1, 4, src_img.size()[2], src_img.size()[3], 3)).to(DEVICE)
        new_src_img = torch.cat((src_img[:,1:,:,:], blank), dim=1) #31,300,186,3
        for i in range(1,max_length+1):
            if i==1:
                tgt_input = tgt_pos[:i, :]
                tgt_img_input = tgt_img[:, :i, :, :, :]
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_pos, tgt_input, self.PAD_IDX)
                if self.args.use_threedimension == 'True':
                    src_pos_2d, tgt_input_2d = self.generate3DInput(tgt_input, src_pos)
                else:
                    src_pos_2d, tgt_input_2d = self.generate2DInput(tgt_input, src_pos)

                logits = self.model(src_pos_2d.float(), tgt_input_2d.float(),  # src_pos, tgt_input,
                                    src_img, tgt_img_input,
                                    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
                _, predicted = torch.max(logits[-1,:,:], 1)
                if i<length:
                    tgt_out = tgt_pos[i, :]
                    LOSS[i-1][0] = self.loss_fn(logits[-1,:,:].reshape(-1, logits[-1,:,:].shape[-1]), tgt_out.reshape(-1).long())
                    loss += self.loss_fn(logits[-1,:,:].reshape(-1, logits[-1,:,:].shape[-1]), tgt_out.reshape(-1).long())

                GAZE[i-1][0] = predicted
                LOGITS[i-1,:] = self.norm(logits[-1,:,:]).reshape(1,-1)
                
                next_tgt_img_input = torch.cat((tgt_img_input, new_src_img[:, predicted, :, :, :]), dim=1)
                next_tgt_input = torch.cat((tgt_input, predicted.view(-1, 1)), dim=0)
            else:
                tgt_input = next_tgt_input
                tgt_img_input = next_tgt_img_input
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_pos, tgt_input, self.PAD_IDX)
                if self.args.use_threedimension == 'True':
                    src_pos_2d, tgt_input_2d = self.generate3DInput(tgt_input, src_pos)
                else:
                    src_pos_2d, tgt_input_2d = self.generate2DInput(tgt_input, src_pos)
                logits = self.model(src_pos_2d.float(), tgt_input_2d.float(),  # src_pos, tgt_input,
                                    src_img, tgt_img_input,
                                    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
                _, predicted = torch.max(logits[-1,:,:], 1)
                if i<length:
                    tgt_out = tgt_pos[i, :]
                    LOSS[i-1][0] = self.loss_fn(logits[-1,:,:].reshape(-1, logits[-1,:,:].shape[-1]), tgt_out.reshape(-1).long())
                    loss += self.loss_fn(logits[-1,:,:].reshape(-1, logits[-1,:,:].shape[-1]), tgt_out.reshape(-1).long())
                GAZE[i-1][0] = predicted
                if self.EOS_IDX in GAZE[:,0] and i >= length:
                    break
                LOGITS[i-1,:] = self.norm(logits[-1,:,:]).reshape(1,-1)
                next_tgt_img_input = torch.cat((next_tgt_img_input, new_src_img[:, predicted, :, :, :]), dim=1)
                next_tgt_input = torch.cat((next_tgt_input, predicted.view(-1,1)), dim=0)
        loss = loss / (length-1)
        if self.EOS_IDX in GAZE:
            endIndex = torch.where(GAZE == self.EOS_IDX)[0][0]
            GAZE = GAZE[:endIndex]
            LOGITS = LOGITS[:endIndex]
        return loss, LOSS, GAZE,LOGITS

    def test_expect(self,src_pos, src_img, tgt_pos, tgt_img):
        tgt_input = tgt_pos[:-1, :]
        tgt_img = tgt_img[:, :-1, :, :, :]
        length = tgt_pos.size(0)
        loss = 0
        max_length = 16
        blank = torch.zeros((1, 4, src_img.size()[2], src_img.size()[3], 3)).to(DEVICE)
        new_src_img = torch.cat((src_img[:,1:,:,:], blank), dim=1) #31,300,186,3
        iter = self.args.stochastic_iteration
        GAZE = torch.zeros((max_length, iter))-1
        for n in range(iter):
            loss_per = 0
            for i in range(1,max_length+1):
                if i==1:
                    tgt_input = tgt_pos[:i, :]
                    tgt_img_input = tgt_img[:, :i, :, :, :]
                    # src: 15, b; tgt_input: 14, b; src_msk: 15, 15; tgt_msk: 13, 13; tgt_padding_msk: 2, 13; src_padding_msk: 2, 15
                    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_pos, tgt_input, self.PAD_IDX)
                    if self.args.use_threedimension == 'True':
                        src_pos_2d, tgt_input_2d = self.generate3DInput(tgt_input, src_pos)
                    else:
                        src_pos_2d, tgt_input_2d = self.generate2DInput(tgt_input, src_pos)
                
                    logits = self.model(src_pos_2d.float(), tgt_input_2d.float(),  # src_pos, tgt_input,
                                        src_img, tgt_img_input,
                                        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
                    logits_new = F.softmax(logits[-1,:,:].view(-1), dim=0)
                    predicted = torch.multinomial(logits_new,1,replacement=True)
                    GAZE[i-1][n] = predicted
                    
                    if i<length:
                        tgt_out = tgt_pos[i, :]
                        loss_per += self.loss_fn(logits[-1,:,:].reshape(-1, logits[-1,:,:].shape[-1]), tgt_out.reshape(-1).long())
                    next_tgt_img_input = torch.cat((tgt_img_input, new_src_img[:, predicted, :, :, :]), dim=1)
                    next_tgt_input = torch.cat((tgt_input, predicted.view(-1, 1)), dim=0)
                
                else:
                    tgt_input = next_tgt_input
                    tgt_img_input = next_tgt_img_input
                    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_pos, tgt_input, self.PAD_IDX)
                    if self.args.use_threedimension == 'True':
                        src_pos_2d, tgt_input_2d = self.generate3DInput(tgt_input, src_pos)
                    else:
                        src_pos_2d, tgt_input_2d = self.generate2DInput(tgt_input, src_pos)
                    logits = self.model(src_pos_2d.float(), tgt_input_2d.float(),  # src_pos, tgt_input,
                                        src_img, tgt_img_input,
                                        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
                    logits_new = F.softmax(logits[-1,:,:].view(-1), dim=0)
                    predicted = torch.multinomial(logits_new,1,replacement=True)
                    GAZE[i-1][n] = predicted
                    if self.EOS_IDX in GAZE[:, n] and i >= length:
                        break
                    if i<length:
                        tgt_out = tgt_pos[i, :]
                        loss_per += self.loss_fn(logits[-1,:,:].reshape(-1, logits[-1,:,:].shape[-1]), tgt_out.reshape(-1).long())
                    next_tgt_img_input = torch.cat((next_tgt_img_input, new_src_img[:, predicted, :, :, :]), dim=1)
                    next_tgt_input = torch.cat((next_tgt_input, predicted.view(-1,1)), dim=0)
                    
            loss += loss_per / (length-1)
        loss= loss / iter
        GAZE_ALL = []
        for i in range(iter):
            if self.EOS_IDX in GAZE[:,i]:
                j = torch.where(GAZE[:,i]==self.EOS_IDX)[0][0]
                GAZE_ALL.append(GAZE[:j, i])
            else:
                GAZE_ALL.append(GAZE[:,i])
        return loss,GAZE_ALL

    def test_gt(self,src_pos, src_img, tgt_pos, tgt_img):
        tgt_input = tgt_pos[:-1, :]
        tgt_img_input = tgt_img[:, :-1, :, :, :]
        length = tgt_pos.size(0) - 1
        soft = torch.nn.Softmax(dim=2)
        # src: 15, b; tgt_input: 14, b; src_msk: 15, 15; tgt_msk: 13, 13; tgt_padding_msk: 2, 13; src_padding_msk: 2, 15
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_pos, tgt_input, self.PAD_IDX)
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
        LOGITS_tf=soft(logits).squeeze(1)
        print(predicted.view(-1))
        return loss,predicted[:-1],tgt_out[:-1],LOGITS_tf[:-1]
        
    def test_step(self, batch, batch_idx):
        src_pos, src_img, tgt_pos, tgt_img = batch
        src_pos = src_pos.to(DEVICE)
        src_img = src_img.to(DEVICE)
        tgt_pos = tgt_pos.to(DEVICE)
        tgt_img = tgt_img.to(DEVICE)

        loss_max, LOSS, GAZE,LOGITS = self.test_max(src_pos, src_img, tgt_pos, tgt_img)
        loss_expect,GAZE_expect = self.test_expect(src_pos, src_img, tgt_pos, tgt_img)
        loss_gt,GAZE_tf,GAZE_gt,LOGITS_tf = self.test_gt(src_pos, src_img, tgt_pos, tgt_img)
        self.log('testing_loss', loss_max, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        if self.args.write_output == 'True':
            return {'loss_max': loss_max, 'LOSS': LOSS, 'GAZE': GAZE, 'LOGITS': LOGITS, 'GAZE_tf': GAZE_tf,
                    'GAZE_gt': GAZE_gt, 'LOGITS_tf': LOGITS_tf, 'GAZE_expect': GAZE_expect}
        else:
            return {'loss_max': loss_max, 'loss_expect': loss_expect, 'loss_gt': loss_gt}

    def test_epoch_end(self, test_step_outputs):
        if self.args.write_output == 'True':
            all_loss, all_gaze, all_gaze_tf, all_gaze_gt, all_logits, all_logits_tf, all_gaze_expect = \
                pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            for output in test_step_outputs:
                #losses = output['LOSS'].cpu().detach().numpy().T
                #all_loss = pd.concat([all_loss, pd.DataFrame(losses)],axis=0)
                gazes = output['GAZE'].cpu().detach().numpy().T
                all_gaze = pd.concat([all_gaze, pd.DataFrame(gazes)],axis=0)
                #logits = output['LOGITS'].cpu().detach().view(1, -1).numpy()
                #all_logits = pd.concat([all_logits, pd.DataFrame(logits)],axis=0)
                gazes_tf = output['GAZE_tf'].cpu().detach().numpy().T
                all_gaze_tf = pd.concat([all_gaze_tf, pd.DataFrame(gazes_tf)],axis=0)
                gazes_gt = output['GAZE_gt'].cpu().detach().numpy().T
                all_gaze_gt = pd.concat([all_gaze_gt, pd.DataFrame(gazes_gt)],axis=0)
                #logits_tf = output['LOGITS_tf'].cpu().detach().view(1, -1).numpy()
                #all_logits_tf = pd.concat([all_logits_tf, pd.DataFrame(logits_tf)],axis=0)
                for i in range(self.args.stochastic_iteration):
                    gazes_expect = output['GAZE_expect'][i].cpu().detach().view(1, -1).numpy()
                    all_gaze_expect = pd.concat([all_gaze_expect, pd.DataFrame(gazes_expect)],axis=0)

            #all_loss.reset_index().drop(['index'],axis=1)
            all_gaze.reset_index().drop(['index'],axis=1)
            #all_logits.reset_index().drop(['index'],axis=1)
            all_gaze_tf.reset_index().drop(['index'],axis=1)
            all_gaze_gt.reset_index().drop(['index'],axis=1)
            #all_logits_tf.reset_index().drop(['index'],axis=1)
            all_gaze_expect.reset_index().drop(['index'],axis=1)
            #all_loss.to_csv('../dataset/checkEvaluation/loss_max.csv', index=False)
            all_gaze.to_csv(self.args.output_path + '/gaze_max' + self.args.output_postfix + '.csv', index=False)
            #all_logits.to_csv('../dataset/checkEvaluation/logits_max.csv', index=False)
            all_gaze_tf.to_csv(self.args.output_path + '/gaze_tf' + self.args.output_postfix + '.csv', index=False)
            all_gaze_gt.to_csv(self.args.output_path + '/gaze_gt' + self.args.output_postfix + '.csv', index=False)
            #all_logits_tf.to_csv('../dataset/checkEvaluation/logits_tf.csv', index=False)
            all_gaze_expect.to_csv(self.args.output_path + '/gaze_expect' + self.args.output_postfix + '.csv', index=False)
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

