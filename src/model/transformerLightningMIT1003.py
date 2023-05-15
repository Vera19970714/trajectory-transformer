import pytorch_lightning as pl
from .models_mit1003 import *
from .models import create_mask
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import pandas as pd
import numpy as np
from evaluation.evaluation_mit1003 import EvaluationMetric
from model.utilis import Sampler


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)

class TransformerModelMIT1003(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.enableLogging = args.enable_logging
        partitionGrid = args.grid_partition
        if partitionGrid != -1:
            self.package_size = int(partitionGrid * partitionGrid)
            self.numOfRegion = args.grid_partition
        else:
            self.package_size = 28
            centerModeFilePath = self.args.data_folder_path + 'centerModeIndex.txt'
            file = open(centerModeFilePath, mode='r')
            centerModeIndex = file.read()
            file.close()
            list1 = centerModeIndex.split('\n')[1:]
            self.centerModeIndexDict = {}
            for a in list1:
                x = a.split(',')
                index = int(x[0])
                x_range = [int(x[1][1]), int(x[1][3])]
                y_range = [int(x[2][1]), int(x[2][3])]
                coor = [int(x[3][1]), int(x[3][3])]
                self.centerModeIndexDict[index] = {'x_range': x_range, 'y_range': y_range, 'coor': coor}

        self.max_length = 10
        self.max_length_total = 1
        torch.manual_seed(0)
        SRC_VOCAB_SIZE = self.package_size + 3
        TGT_VOCAB_SIZE = self.package_size + 3
        self.PAD_IDX = self.package_size
        self.BOS_IDX = self.package_size + 1
        self.EOS_IDX = self.package_size + 2
        self.numofextraindex = 3
        EMB_SIZE = 512
        NHEAD = 4
        FFN_HID_DIM = 512
        NUM_ENCODER_LAYERS = 4
        NUM_DECODER_LAYERS = 4
        inputDim = 2
        add_salient_OD = args.add_salient_OD
        if args.add_salient_OD == 'True':
            add_salient_OD = True
        elif args.add_salient_OD == 'False':
            add_salient_OD = False

        if args.feature_extractor == 'CNN':
            isCNNExtractor = True
        elif args.feature_extractor == 'LP':
            isCNNExtractor = False
        else:
            print('Wrong value of args.feature_extractor')
        if args.decoder_input == 'index':
            isDecoderOutputFea = False
        elif args.decoder_input == 'plus_feature':
            isDecoderOutputFea = True
        else:
            print('Wrong value of args.decoder_input')
            quit()
        if args.global_token == 'True':
            isGlobalToken = True
        elif args.global_token == 'False':
            isGlobalToken = False
        else:
            print('Wrong value of args.decoder_input')
            quit()
        self.isGlobalToken = isGlobalToken
        self.model = Seq2SeqTransformer4MIT1003(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                        NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, inputDim, FFN_HID_DIM,
                                                isCNNExtractor, isDecoderOutputFea, isGlobalToken,
                                                add_salient_OD).to(DEVICE).float()
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.PAD_IDX)
        self.norm = torch.nn.Softmax(dim=1)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, betas=(0.9, 0.98), eps=1e-9)
        if self.enableLogging == 'True':
            self.loggerS = SummaryWriter(f'./lightning_logs/{args.log_dir}')
        self.total_step = 0
        self.metrics = EvaluationMetric(trainingGrid=args.grid_partition)
        self.sampler = Sampler()

    def log_gradients_in_model(self, step):
        for tag, value in self.model.named_parameters():
            # print('-'*10)
            if value.grad is not None:
                # print(tag, value.grad.cpu())
                self.loggerS.add_histogram(tag + "/grad", value.grad.cpu(), step)
            # print('-' * 10)

    def training_step(self, batch, batch_idx):
        src_pos, src_img, tgt_pos, tgt_img = batch
        # src_pos(28, b), src_img(b, 28, w, h, 3), tgt_pos(max_len, b), tgt_img(b, max_len, w, h, 3)
        src_pos.to(DEVICE)
        #src_img = src_img.to(DEVICE)
        tgt_pos.to(DEVICE)
        #tgt_img = tgt_img.to(DEVICE)

        src_pos_2d, tgt_input_2d, src_img, tgt_img, src_mask, tgt_mask, \
        src_padding_mask, tgt_padding_mask, src_padding_mask = self.processData2d(src_pos, src_img, tgt_pos,
                                                                                  tgt_img)

        logits = self.model(src_pos_2d.float(), tgt_input_2d.float(),  # src_pos, tgt_input,
                            src_img, tgt_img,
                            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        tgt_out = tgt_pos[1:, :]
        loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
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


    def processData2d(self, src_pos, src_img, tgt_pos, tgt_img):
        for i in range(len(src_img)):
            src_img[i].to(DEVICE)
            tgt_img[i].to(DEVICE)
            tgt_img[i] = tgt_img[i][:-1, :, :, :]

        tgt_input = tgt_pos[:-1, :]
        #tgt_img = tgt_img[:, :-1, :, :, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_pos, tgt_input, self.PAD_IDX,
                                                                             self.isGlobalToken)
        if self.args.grid_partition != -1:
            src_pos_2d, tgt_input_2d = self.generate2DInput(tgt_input, src_pos)
        else:
            src_pos_2d, tgt_input_2d = self.generate2DInputCenterMode(tgt_input, src_pos)

        return src_pos_2d, tgt_input_2d, src_img, tgt_img, src_mask, tgt_mask, \
               src_padding_mask, tgt_padding_mask, src_padding_mask

    def generate2DInput(self, tgt_input, src_pos):
        tgt_input_2d = torch.zeros((tgt_input.size()[0], tgt_input.size()[1], 2)).to(DEVICE).float()
        tgt_input_2d[:, :, 0] = tgt_input // self.numOfRegion #/ (self.numOfRegion-1) # todo: all changed to divide by 4
        tgt_input_2d[:, :, 1] = torch.remainder(tgt_input, self.numOfRegion) #/ (self.numOfRegion-1)
        tgt_input_2d[0, :, 0] = (self.numOfRegion-1)/2 #/ (self.numOfRegion-1)
        tgt_input_2d[0, :, 1] = (self.numOfRegion-1)/2 #/ (self.numOfRegion-1)

        src_pos_2d = torch.zeros((src_pos.size()[0], src_pos.size()[1], 2)).to(DEVICE).float()
        src_pos_2d[:, :, 0] = src_pos // self.numOfRegion #/ (self.numOfRegion-1)
        src_pos_2d[:, :, 1] = torch.remainder(src_pos, self.numOfRegion) #/ (self.numOfRegion-1)
        src_pos_2d[0, :, 0] = -1
        src_pos_2d[0, :, 1] = -1
        #src_pos_2d[0, :, 0] = (self.numOfRegion - 1) / 2 / (self.numOfRegion-1)
        #src_pos_2d[0, :, 1] = (self.numOfRegion - 1) / 2 / (self.numOfRegion-1)

        return src_pos_2d, tgt_input_2d

    '''def generate2DInputCenterMode(self, tgt_input, src_pos):
        tgt_input_2d = torch.zeros((tgt_input.size()[0], tgt_input.size()[1], 2)).to(DEVICE).float()
        for i in range(tgt_input.size()[0]):
            for j in range(tgt_input.size()[1]):
                if tgt_input[i][j] < self.package_size:
                    index = self.centerModeIndexDict[tgt_input[i][j].item()]['coor']
                    tgt_input_2d[i][j][0] = index[0] / 8
                    tgt_input_2d[i][j][1] = index[1] / 8
                else:
                    tgt_input_2d[i][j][0] = -1
                    tgt_input_2d[i][j][1] = -1

        src_pos_2d = torch.zeros((src_pos.size()[0], src_pos.size()[1], 2)).to(DEVICE).float()
        for i in range(src_pos.size()[0]):
            for j in range(src_pos.size()[1]):
                if src_pos[i][j] < self.package_size:
                    index = self.centerModeIndexDict[src_pos[i][j].item()]['coor']
                    src_pos_2d[i][j][0] = index[0] / 8
                    src_pos_2d[i][j][1] = index[1] / 8
                else:
                    src_pos_2d[i][j][0] = -1
                    src_pos_2d[i][j][1] = -1

        return src_pos_2d, tgt_input_2d'''


    def validation_step(self, batch, batch_idx):
        src_pos, src_img, tgt_pos, tgt_img = batch
        # src_pos(28, b), src_img(b, 28, w, h, 3), tgt_pos(max_len, b), tgt_img(b, max_len, w, h, 3)
        src_pos.to(DEVICE)
        #src_img = src_img.to(DEVICE)
        tgt_pos.to(DEVICE)
        #tgt_img = tgt_img.to(DEVICE)

        src_pos_2d, tgt_input_2d, src_img, tgt_img, src_mask, tgt_mask, \
            src_padding_mask, tgt_padding_mask, src_padding_mask = self.processData2d(src_pos, src_img, tgt_pos,
                                                                                      tgt_img)

        logits = self.model(src_pos_2d.float(), tgt_input_2d.float(),  # src_pos, tgt_input,
                            src_img, tgt_img,
                            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        # logits: 11, 1, 31, tgt_out: 11, 1
        tgt_out = tgt_pos[1:, :]
        loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        _, predicted = torch.max(logits, 2)
        #print(predicted.view(1, -1))

        # calculate SED and SBTDE
        b = tgt_out.size()[1]
        sed = []
        sbtde = []
        lenScanpath = tgt_out.size()[0]
        returnSED = False
        if lenScanpath >= self.metrics.minLen:
            returnSED = True
            for index in range(b):
                scanpath_gt = tgt_out[:self.metrics.minLen, index].detach().cpu().numpy()
                scanpath_pre = predicted[:self.metrics.minLen, index].detach().cpu().numpy()
                sed_i, sbtde_i = self.metrics.get_sed_and_sbtde(scanpath_gt, scanpath_pre)
                sed.append(sed_i)
                sbtde.append(sbtde_i)
            sed = np.mean(sed)
            sbtde = np.mean(sbtde)

        if self.enableLogging == 'True':
            self.log('validation_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        if returnSED:
            return {'loss': loss, 'sed': sed, 'sbtde': sbtde}
        else:
            return {'loss': loss, 'sed': -1, 'sbtde': -1}

    def validation_epoch_end(self, validation_step_outputs):
        avg_loss = torch.stack([x['loss'] for x in validation_step_outputs]).mean()
        if self.enableLogging == 'True':
            self.log('validation_loss_each_epoch', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        loss_sed = []
        loss_sbtde = []
        for x in validation_step_outputs:
            if x['sed'] != -1:
                loss_sed.append(x['sed'])
                loss_sbtde.append(x['sbtde'])
        if len(loss_sed) != 0:
            avg_loss_sed = np.mean(loss_sed)
            avg_loss_sbtde = np.mean(loss_sbtde)
            print('validation_loss_each_epoch: ', avg_loss, ', sed: ', avg_loss_sed, ', sbtde: ', avg_loss_sbtde)
            if self.enableLogging == 'True':
                self.log('validation_evaluation_sed', avg_loss_sed, on_step=False, on_epoch=True, prog_bar=True,
                         sync_dist=True)
                self.log('validation_evaluation_sbtde', avg_loss_sbtde, on_step=False, on_epoch=True, prog_bar=True,
                         sync_dist=True)
                self.log('validation_evaluation_all', (avg_loss_sed+avg_loss_sbtde), on_step=False, on_epoch=True, prog_bar=True,
                         sync_dist=True)

    def test_max(self, src_pos, src_img, tgt_pos, tgt_img):
        # If target sequence length less than 10, then skip this function
        gt_seq = tgt_pos[1:, :]
        tgt_seq_len = gt_seq.size()[0]
        if tgt_seq_len - 1 < self.metrics.minLen:
            return -1, -1, -1, -1
        gt_seq = gt_seq[:self.metrics.minLen, :]
        tgt_input = tgt_pos[:-1, :]
        tgt_img = tgt_img[:, :-1, :, :, :]
        length = tgt_pos.size(0)
        loss = 0
        LOSS = torch.zeros((length - 1, 1)) - 1
        GAZE = torch.zeros((self.max_length, 1)) - 1
        LOGITS = torch.zeros((self.max_length, self.package_size+self.numofextraindex))
        blank = torch.zeros((1, self.numofextraindex, src_img.size()[2], src_img.size()[3], src_img.size()[4])).to(DEVICE)
        new_src_img = torch.cat((src_img, blank), dim=1)  # 31,300,186,3
        for i in range(1, self.max_length + 1):
            if i == 1:
                tgt_input = tgt_pos[:i, :]
                tgt_img_input = tgt_img[:, :i, :, :, :]
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_pos, tgt_input, self.PAD_IDX,
                                                                                     self.isGlobalToken)
                if self.args.grid_partition != -1:
                    src_pos_2d, tgt_input_2d = self.generate2DInput(tgt_input, src_pos)
                else:
                    src_pos_2d, tgt_input_2d = self.generate2DInputCenterMode(tgt_input, src_pos)
                logits = self.model(src_pos_2d.float(), tgt_input_2d.float(),  # src_pos, tgt_input,
                                    src_img.float(), tgt_img_input.float(),
                                    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
                _, predicted = torch.max(logits[-1, :, :], 1)
                # test random results
                predicted = torch.randint(16, (1,)).to(DEVICE)
                if i < length:
                    tgt_out = tgt_pos[i, :]
                    LOSS[i - 1][0] = self.loss_fn(logits[-1, :, :].reshape(-1, logits[-1, :, :].shape[-1]),
                                                  tgt_out.reshape(-1).long())
                    loss += self.loss_fn(logits[-1, :, :].reshape(-1, logits[-1, :, :].shape[-1]),
                                         tgt_out.reshape(-1).long())
                GAZE[i - 1][0] = predicted
                LOGITS[i - 1, :] = self.norm(logits[-1, :, :]).reshape(1, -1)
                next_tgt_img_input = torch.cat((tgt_img_input, new_src_img[:, predicted, :, :, :]), dim=1)
                next_tgt_input = torch.cat((tgt_input, predicted.view(-1, 1)), dim=0)
            else:
                tgt_input = next_tgt_input
                tgt_img_input = next_tgt_img_input
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_pos, tgt_input, self.PAD_IDX,
                                                                                     self.isGlobalToken)
                if self.args.grid_partition != -1:
                    src_pos_2d, tgt_input_2d = self.generate2DInput(tgt_input, src_pos)
                else:
                    src_pos_2d, tgt_input_2d = self.generate2DInputCenterMode(tgt_input, src_pos)
                logits = self.model(src_pos_2d.float(), tgt_input_2d.float(),  # src_pos, tgt_input,
                                    src_img.float(), tgt_img_input.float(),
                                    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
                _, predicted = torch.max(logits[-1, :, :], 1)
                # test random results
                predicted = torch.randint(16, (1,)).to(DEVICE)
                if i < length:
                    tgt_out = tgt_pos[i, :]
                    LOSS[i - 1][0] = self.loss_fn(logits[-1, :, :].reshape(-1, logits[-1, :, :].shape[-1]),
                                                  tgt_out.reshape(-1).long())
                    loss += self.loss_fn(logits[-1, :, :].reshape(-1, logits[-1, :, :].shape[-1]),
                                         tgt_out.reshape(-1).long())
                GAZE[i - 1][0] = predicted
                # COMMENT these because the rules have been changed, the output length is always 10
                #if self.EOS_IDX in GAZE[:, 0] and i >= length:
                #    break
                LOGITS[i - 1, :] = self.norm(logits[-1, :, :]).reshape(1, -1)
                next_tgt_img_input = torch.cat((next_tgt_img_input, new_src_img[:, predicted, :, :, :]), dim=1)
                next_tgt_input = torch.cat((next_tgt_input, predicted.view(-1, 1)), dim=0)
        loss = loss / (length - 1)
        # compare gt_seq with GAZE and compute SED, they should be of same size
        b = gt_seq.size()[1]
        sed = []
        sbtde = []
        for index in range(b):
            scanpath_gt = gt_seq[:self.metrics.minLen, index].detach().cpu().numpy()
            scanpath_pre = GAZE[:self.metrics.minLen, index].detach().cpu().numpy()
            sed_i, sbtde_i = self.metrics.get_sed_and_sbtde(scanpath_gt, scanpath_pre)
            sed.append(sed_i)
            sbtde.append(sbtde_i)
        sed = np.mean(sed)
        sbtde = np.mean(sbtde)
        return sed, sbtde, gt_seq, GAZE
        # COMMENT these because the rules have been changed, the output length is always 10
        '''if self.EOS_IDX in GAZE:
            endIndex = torch.where(GAZE == self.EOS_IDX)[0][0]
            GAZE = GAZE[:endIndex]
            LOGITS = LOGITS[:endIndex]
        return loss, LOSS, GAZE, LOGITS'''

    '''def test_saliency_max(self, imgSize, src_pos, src_img, tgt_pos, tgt_img, scanpath):
        # If target sequence length less than 10, then skip this function
        gt_seq = tgt_pos[1:, :]
        tgt_seq_len = gt_seq.size()[0]
        tgt_input = tgt_pos[:-1, :]
        tgt_img = tgt_img[:, :-1, :, :, :]
        length = tgt_pos.size(0)
        print(length)
        
        loss = 0
        LOSS = torch.zeros((length - 1, 1)) - 1
        GAZE = torch.zeros((self.max_length_total, 1)) - 1
        LOGITS = torch.zeros((self.max_length_total, self.package_size+self.numofextraindex))
        blank = torch.zeros((1, self.numofextraindex, src_img.size()[2], src_img.size()[3], 3)).to(DEVICE)
        new_src_img = torch.cat((src_img, blank), dim=1)  # 31,300,186,3
        for i in range(1, self.max_length_total + 1):
            if i == 1:
                tgt_input = tgt_pos[:i, :]
                tgt_img_input = tgt_img[:, :i, :, :, :]
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_pos, tgt_input, self.PAD_IDX,
                                                                                     self.isGlobalToken)
                if self.args.grid_partition != -1:
                    src_pos_2d, tgt_input_2d = self.generate2DInput(tgt_input, src_pos)
                else:
                    src_pos_2d, tgt_input_2d = self.generate2DInputCenterMode(tgt_input, src_pos)
                logits = self.model(src_pos_2d.float(), tgt_input_2d.float(),  # src_pos, tgt_input,
                                    src_img, tgt_img_input,
                                    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
                _, predicted = torch.max(logits[-1, :, :], 1)
                if i < length:
                    tgt_out = tgt_pos[i, :]
                    LOSS[i - 1][0] = self.loss_fn(logits[-1, :, :].reshape(-1, logits[-1, :, :].shape[-1]),
                                                  tgt_out.reshape(-1).long())
                    loss += self.loss_fn(logits[-1, :, :].reshape(-1, logits[-1, :, :].shape[-1]),
                                         tgt_out.reshape(-1).long())
                GAZE[i - 1][0] = predicted
                LOGITS[i - 1, :] = self.norm(logits[-1, :, :]).reshape(1, -1)
                next_tgt_img_input = torch.cat((tgt_img_input, new_src_img[:, predicted, :, :, :]), dim=1)
                next_tgt_input = torch.cat((tgt_input, predicted.view(-1, 1)), dim=0)
            else:
                tgt_input = next_tgt_input
                tgt_img_input = next_tgt_img_input
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_pos, tgt_input, self.PAD_IDX,
                                                                                     self.isGlobalToken)
                if self.args.grid_partition != -1:
                    src_pos_2d, tgt_input_2d = self.generate2DInput(tgt_input, src_pos)
                else:
                    src_pos_2d, tgt_input_2d = self.generate2DInputCenterMode(tgt_input, src_pos)
                logits = self.model(src_pos_2d.float(), tgt_input_2d.float(),  # src_pos, tgt_input,
                                    src_img, tgt_img_input,
                                    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
                _, predicted = torch.max(logits[-1, :, :], 1)
                if self.EOS_IDX in GAZE[:, 0] and i >= length:
                    break
                if i < length:
                    tgt_out = tgt_pos[i, :]
                    LOSS[i - 1][0] = self.loss_fn(logits[-1, :, :].reshape(-1, logits[-1, :, :].shape[-1]),
                                                  tgt_out.reshape(-1).long())
                    loss += self.loss_fn(logits[-1, :, :].reshape(-1, logits[-1, :, :].shape[-1]),
                                         tgt_out.reshape(-1).long())
                GAZE[i - 1][0] = predicted
                # COMMENT these because the rules have been changed, the output length is always 10
                if self.EOS_IDX in GAZE[:, 0] and i >= length:
                   break
                LOGITS[i - 1, :] = self.norm(logits[-1, :, :]).reshape(1, -1)
                next_tgt_img_input = torch.cat((next_tgt_img_input, new_src_img[:, predicted, :, :, :]), dim=1)
                next_tgt_input = torch.cat((next_tgt_input, predicted.view(-1, 1)), dim=0)
        loss = loss / (length - 1)
        if self.EOS_IDX in GAZE:
            endIndex = torch.where(GAZE == self.EOS_IDX)[0][0]
            GAZE = GAZE[:endIndex]
            LOGITS = LOGITS[:endIndex]
        
        # compare gt_seq with GAZE and compute SED, they should be of same size
        imgSize = imgSize.detach().cpu().numpy()
        auc,nss = self.metrics.saliencyEvaluation(scanpath.detach().cpu().numpy(),GAZE.detach().cpu().numpy(),imgSize[0],imgSize[1])
        return auc, nss
        # COMMENT these because the rules have been changed, the output length is always 10
        # return loss, LOSS, GAZE, LOGITS'''

    
    '''def test_expect(self, src_pos, src_img, tgt_pos, tgt_img):
        # TODO: remain unchanged, need to change based on new free viewing dataset
        tgt_input = tgt_pos[:-1, :]
        tgt_img = tgt_img[:, :-1, :, :, :]
        length = tgt_pos.size(0)
        loss = 0
        max_length = 16
        blank = torch.zeros((1, self.numofextraindex, src_img.size()[2], src_img.size()[3], 3)).to(DEVICE)
        new_src_img = torch.cat((src_img, blank), dim=1)  # 31,300,186,3
        iter = self.args.stochastic_iteration
        GAZE = torch.zeros((max_length, iter)) - 1
        for n in range(iter):
            loss_per = 0
            for i in range(1, max_length + 1):
                if i == 1:
                    tgt_input = tgt_pos[:i, :]
                    tgt_img_input = tgt_img[:, :i, :, :, :]
                    # src: 15, b; tgt_input: 14, b; src_msk: 15, 15; tgt_msk: 13, 13; tgt_padding_msk: 2, 13; src_padding_msk: 2, 15
                    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_pos, tgt_input,
                                                                                         self.PAD_IDX)
                    src_pos_2d, tgt_input_2d = self.generate2DInput(tgt_input, src_pos)

                    logits = self.model(src_pos_2d.float(), tgt_input_2d.float(),  # src_pos, tgt_input,
                                        src_img, tgt_img_input,
                                        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
                    logits_new = F.softmax(logits[-1, :, :].view(-1), dim=0)
                    predicted = torch.multinomial(logits_new, 1, replacement=True)
                    GAZE[i - 1][n] = predicted

                    if i < length:
                        tgt_out = tgt_pos[i, :]
                        loss_per += self.loss_fn(logits[-1, :, :].reshape(-1, logits[-1, :, :].shape[-1]),
                                                 tgt_out.reshape(-1).long())
                    next_tgt_img_input = torch.cat((tgt_img_input, new_src_img[:, predicted, :, :, :]), dim=1)
                    next_tgt_input = torch.cat((tgt_input, predicted.view(-1, 1)), dim=0)

                else:
                    tgt_input = next_tgt_input
                    tgt_img_input = next_tgt_img_input
                    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_pos, tgt_input,
                                                                                         self.PAD_IDX)
                    src_pos_2d, tgt_input_2d = self.generate2DInput(tgt_input, src_pos)
                    logits = self.model(src_pos_2d.float(), tgt_input_2d.float(),  # src_pos, tgt_input,
                                        src_img, tgt_img_input,
                                        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
                    logits_new = F.softmax(logits[-1, :, :].view(-1), dim=0)
                    predicted = torch.multinomial(logits_new, 1, replacement=True)
                    GAZE[i - 1][n] = predicted
                    if self.EOS_IDX in GAZE[:, n] and i >= length:
                        break
                    if i < length:
                        tgt_out = tgt_pos[i, :]
                        loss_per += self.loss_fn(logits[-1, :, :].reshape(-1, logits[-1, :, :].shape[-1]),
                                                 tgt_out.reshape(-1).long())
                    next_tgt_img_input = torch.cat((next_tgt_img_input, new_src_img[:, predicted, :, :, :]), dim=1)
                    next_tgt_input = torch.cat((next_tgt_input, predicted.view(-1, 1)), dim=0)

            loss += loss_per / (length - 1)
        loss = loss / iter
        GAZE_ALL = []
        for i in range(iter):
            if self.EOS_IDX in GAZE[:, i]:
                j = torch.where(GAZE[:, i] == self.EOS_IDX)[0][0]
                GAZE_ALL.append(GAZE[:j, i])
            else:
                GAZE_ALL.append(GAZE[:, i])
        return loss, GAZE_ALL'''

    '''def test_gt(self, src_pos, src_img, tgt_pos, tgt_img):
        
        NOT USED IN THE ACTUALLY EVALUATION
        :param src_pos:
        :param src_img:
        :param tgt_pos:
        :param tgt_img:
        :return:
        
        tgt_input = tgt_pos[:-1, :]
        tgt_img_input = tgt_img[:, :-1, :, :, :]
        length = tgt_pos.size(0) - 1
        soft = torch.nn.Softmax(dim=2)
        # src: 15, b; tgt_input: 14, b; src_msk: 15, 15; tgt_msk: 13, 13; tgt_padding_msk: 2, 13; src_padding_msk: 2, 15
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_pos, tgt_input, self.PAD_IDX)
        src_pos_2d, tgt_input_2d = self.generate2DInput(tgt_input, src_pos)

        logits = self.model(src_pos_2d.float(), tgt_input_2d.float(),  # src_pos, tgt_input,
                            src_img, tgt_img_input,
                            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        tgt_out = tgt_pos[1:, :]
        loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        _, predicted = torch.max(logits, 2)
        LOGITS_tf = soft(logits).squeeze(1)
        print(predicted.view(-1))
        return loss, predicted[:-1], tgt_out[:-1], LOGITS_tf[:-1]'''

    def test_step(self, batch, batch_idx):
        imageName, src_pos, src_img, tgt_pos, tgt_img = batch
        # src_img and tgt_img always have batch size 1
        src_img = torch.stack(src_img)
        tgt_img = torch.stack(tgt_img)
        #scanpath = torch.squeeze(scanpath)
        src_pos.to(DEVICE)
        src_img = src_img.to(DEVICE)
        tgt_pos.to(DEVICE)
        tgt_img = tgt_img.to(DEVICE)
        #scanpath = scanpath.to(DEVICE)
        #imgSize = imgSize.to(DEVICE)
        sed, sbtde, gt_seq, predicted_seq = self.test_max(src_pos, src_img, tgt_pos, tgt_img)
        '''if self.args.saliency_metric == 'True':
            # NOT TESTED
            auc, nss = self.test_saliency_max(imgSize, src_pos, src_img, tgt_pos, tgt_img,scanpath)
        # TODO: these functions havent changed regard to free viewing datasets
        #loss_expect, GAZE_expect = self.test_expect(src_pos, src_img, tgt_pos, tgt_img)
        #loss_gt, GAZE_tf, GAZE_gt, LOGITS_tf = self.test_gt(src_pos, src_img, tgt_pos, tgt_img)
            if self.enableLogging == 'True' and sed != -1 and sbtde != -1:
                #self.log('testing_loss', loss_max, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log('testing_loss_sed', sed, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log('testing_loss_sbtde', sbtde, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
                # self.log('testing_loss_sed_topk', meanSed_topk, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
                # self.log('testing_loss_sbtde_topk', meanSbtde_topk, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log('testing_loss_auc', auc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log('testing_loss_nss', nss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            return {'testing_sed': sed, 'testing_sbtde': sbtde, 'testing_auc': auc,'testing_nss': nss, 'testing_image': imageName}
        else:'''
        if sed != -1 and sbtde != -1:
            if self.enableLogging == 'True':
                #self.log('testing_loss', loss_max, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log('testing_loss_sed', sed, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log('testing_loss_sbtde', sbtde, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
                # self.log('testing_loss_sed_topk', meanSed_topk, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
                # self.log('testing_loss_sbtde_topk', meanSbtde_topk, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            if self.args.write_output == 'True':
                return {'testing_sed': sed, 'testing_sbtde': sbtde, 'testing_image': imageName, 'gt_gaze': gt_seq, 'pred_gaze': predicted_seq}
            else:
                return {'testing_sed': sed, 'testing_sbtde': sbtde, 'testing_image': imageName}

    def test_epoch_end(self, test_step_outputs):
        if self.args.write_output == 'True':
            all_gaze, all_gaze_gt = pd.DataFrame(), pd.DataFrame()
            for output in test_step_outputs:
                gazes = output['pred_gaze'].cpu().detach().numpy().T.tolist()
                gazes[0].append(output['testing_image'])
                all_gaze = pd.concat([all_gaze, pd.DataFrame(gazes)], axis=0)
                gazes_gt = output['gt_gaze'].cpu().detach().numpy().T.tolist()
                gazes_gt[0].append(output['testing_image'])
                all_gaze_gt = pd.concat([all_gaze_gt, pd.DataFrame(gazes_gt)], axis=0)
                '''for i in range(self.args.stochastic_iteration):
                    gazes_expect = output['GAZE_expect'][i].cpu().detach().view(1, -1).numpy()
                    all_gaze_expect = pd.concat([all_gaze_expect, pd.DataFrame(gazes_expect)], axis=0)'''

            all_gaze.reset_index().drop(['index'], axis=1)
            all_gaze_gt.reset_index().drop(['index'], axis=1)
            all_gaze.to_csv(self.args.output_path + '/gaze_max' + self.args.output_postfix + '.csv', index=False)
            all_gaze_gt.to_csv(self.args.output_path + '/gaze_gt' + self.args.output_postfix + '.csv', index=False)

        # have been changed based on the new evaluation SED
        loss_sed = []
        loss_sbtde = []
        testResult_sed = []
        testResult_sbtde = []

        for x in test_step_outputs:
            if x['testing_sed'] != -1:
                loss_sed.append(x['testing_sed'])
                loss_sbtde.append(x['testing_sbtde'])
                testResult_sed.append((x['testing_image'], x['testing_sed']))
                testResult_sbtde.append((x['testing_image'], x['testing_sbtde']))
        if len(loss_sed) != 0:
            SED, SBTDE, sppSED, sppSBTDE = self.metrics.get_Sed_and_Sbtde(testResult_sed, testResult_sbtde)
            sppSED = np.mean(sppSED)
            sppSBTDE = np.mean(sppSBTDE)
            avg_loss_sed = np.mean(SED)
            avg_loss_sbtde = np.mean(SBTDE)
            print('Evaluation results || SED: ', avg_loss_sed, ', SBTDE: ', avg_loss_sbtde, ', spp SED: ', sppSED, ', spp SBTDE: ', sppSBTDE)
            if self.enableLogging == 'True':
                self.log('testing_evaluation_meanSED', avg_loss_sed, on_step=False, on_epoch=True, prog_bar=True,
                            sync_dist=True)
                self.log('testing_evaluation_meanSBTDE', avg_loss_sbtde, on_step=False, on_epoch=True, prog_bar=True,
                            sync_dist=True)
                self.log('testing_evaluation_sppSED', sppSED, on_step=False, on_epoch=True, prog_bar=True,
                            sync_dist=True)
                self.log('testing_evaluation_sppSBTDE', sppSBTDE, on_step=False, on_epoch=True, prog_bar=True,
                            sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.scheduler_lambda1,
                                                    gamma=self.args.scheduler_lambda2)
        return [optimizer], [scheduler]

