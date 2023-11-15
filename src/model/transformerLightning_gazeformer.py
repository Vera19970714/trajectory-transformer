import pytorch_lightning as pl
from .gazeformer import gazeformer
from .models import *
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import pandas as pd
import numpy as np
import sys
sys.path.append('./src/')
from evaluation.evaluation_model import behavior
from evaluation.saliency_metric import saliency_map_metric, nw_matching, compare_multi_gazes

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TransformerModel_Gazeformer(pl.LightningModule):
    def __init__(self, args, max_len):
        super().__init__()
        self.args = args
        self.max_len = max_len
        torch.manual_seed(0)
        TGT_VOCAB_SIZE = 0
        self.PAD_IDX = self.args.package_size+1
        self.BOS_IDX = self.args.package_size+2
        self.EOS_IDX = self.args.package_size #+3
        self.model = gazeformer(spatial_dim=(20,32), max_len=max_len).to(DEVICE).float()
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.loss_fn = [torch.nn.CrossEntropyLoss(ignore_index=self.PAD_IDX[0]), torch.nn.CrossEntropyLoss(ignore_index=self.PAD_IDX[1])]
        self.norm = torch.nn.Softmax(dim=1)
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, betas=(0.9, 0.98), eps=1e-9)
        self.loggerS= SummaryWriter(f'./lightning_logs/{args.log_name}')
        self.total_step = 0
        self.loss_fn_token = torch.nn.NLLLoss().to(DEVICE)
        self.loss_fn_xy = nn.L1Loss(reduction='mean').to(DEVICE)

    def log_gradients_in_model(self, step):
        for tag, value in self.model.named_parameters():
            #print('-'*10)
            if value.grad is not None:
                #print(tag, value.grad.cpu())
                self.loggerS.add_histogram(tag + "/grad", value.grad.cpu(), step)
            #print('-' * 10)

    def train_one_dataset(self, batch, type, return_gaze=False):
        src_pos, src_img, tgt_pos, tgt_img = batch
        # databuilder output: src 10, 640, 2048; firstfix: 10, 2; task: 10, 768; token_prob: 7,10,2;
        # package_target: 23, 1; src_img: 1, 640, 2048; package_seq: 9, 1; tgt_img: 1, 15, 2048
        src_pos = src_pos.to(DEVICE)
        src_img = src_img.to(DEVICE)
        tgt_pos = tgt_pos.to(DEVICE)
        tgt_img = tgt_img.to(DEVICE)

        src_pos_2d, tgt_input_2d, src_img, tgt_img, src_mask, tgt_mask, \
        src_padding_mask, tgt_padding_mask, src_padding_mask = self.processData3d(src_pos, src_img, tgt_pos,
                                                                                  tgt_img, type)

        # databuilder output: src 10, 640, 2048; firstfix: 10, 2; task: 10, 768; token_prob: 7,10,2;
        # input: first_fix: b, 2, src_img: b, 640, 2048, task_img: b, 15, 2048,
        first_fix = tgt_input_2d[0, :, :2].long()
        end_logits, xs_out, ys_out = self.model(first_fix, src_img, tgt_img)
                            #src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        # output: 10,b,2; 10,b,1; 10,b,1
        gt = tgt_pos[1:]  # 8, 1
        loss = torch.tensor(0).float().to(DEVICE)
        gazes = []
        for i in range(gt.size()[1]):
            gt_ = gt[:, i] #8
            end_token = torch.where(gt_ == self.EOS_IDX[type])[0]
            logit_ = end_logits[:(end_token+1), i, :].float()  # 10, 2
            gt_ = gt_[:end_token]
            gtx = gt_ // self.args.shelf_col[type]
            gty = torch.remainder(gt_, self.args.shelf_col[type])
            xs_ = xs_out[:end_token, i, 0]  # 10,
            ys_ = ys_out[:end_token, i, 0]  # 10
            logit_gt = torch.zeros(((end_token+1))).type(torch.LongTensor).to(DEVICE)
            logit_gt[-1] = 1
            loss += self.loss_fn_token(logit_, logit_gt)
            #xs_, ys_ = torch.clamp(xs_, min=0, max=self.args.shelf_col[type]), torch.clamp(ys_, min=0, max=self.args.shelf_row[type])
            loss += self.loss_fn_xy(xs_, gtx)
            loss += self.loss_fn_xy(ys_, gty)
            if return_gaze:
                end_lo = torch.argmax(end_logits[:, 0, :], axis=-1)[1:] #10, 1, 2
                end_index = torch.where(end_lo == 1)[0]
                if len(end_index) == 0:
                    end_index = end_lo.size()[0]
                else:
                    end_index = end_index[0]
                end_index += 1 # first one cannot be ending
                xs_ = xs_out[:end_index, i, 0]  # 10,
                ys_ = ys_out[:end_index, i, 0]
                xs_ = torch.round(torch.clamp(xs_, min=0, max=self.args.shelf_col[type]).unsqueeze(-1))
                ys_ = torch.round(torch.clamp(ys_, min=0, max=self.args.shelf_row[type]).unsqueeze(-1))
                xy = xs_ * self.args.shelf_col[type] + ys_
                gazes.append(xy)
        if return_gaze:
            return loss, gazes
        return loss

    def training_step(self, batch, batch_idx):
        data1, data2 = batch
        if len(data1) != 0:
            loss1 = self.train_one_dataset(data1, 0)
        else:
            loss1 = 0
        if len(data2) != 0:
            loss2 = self.train_one_dataset(data2, 1)
        else:
            loss2 = 0
        loss = loss1 + loss2
        #self.log_gradients_in_model(self.total_step)
        self.total_step += 1
        self.log('training_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'loss': loss, }

    def training_epoch_end(self, training_step_outputs):
        avg_loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
        self.log('training_loss_each_epoch', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)

    def processData3d(self, src_pos, src_img, tgt_pos, tgt_img, type):
        tgt_input = tgt_pos[:-1, :]
        #tgt_img = tgt_img[:, :-1, :, :, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_pos, tgt_input, self.PAD_IDX[type])

        src_pos_2d, tgt_input_2d = self.generate3DInput(tgt_input, src_pos, type)

        return src_pos_2d, tgt_input_2d,  src_img, tgt_img, src_mask, tgt_mask, \
               src_padding_mask, tgt_padding_mask, src_padding_mask


    def generate3DInput(self, tgt_input, src_pos, type):
        tgt_input_2d = torch.zeros((tgt_input.size()[0], tgt_input.size()[1], 3)).to(DEVICE).float()

        tgt_input_2d[:, :, 0] = tgt_input // self.args.shelf_col[type]
        tgt_input_2d[:, :, 1] = torch.remainder(tgt_input, self.args.shelf_col[type])
        tgt_input_2d[0, :, 0] = float(self.args.shelf_row[type]) / 2
        tgt_input_2d[0, :, 1] = float(self.args.shelf_col[type]) / 2

        src_pos_2d = torch.zeros((src_pos.size()[0], src_pos.size()[1], 3)).to(DEVICE).float()
        src_pos_2d[:, :, 0] = src_pos // self.args.shelf_col[type]
        src_pos_2d[:, :, 1] = torch.remainder(src_pos, self.args.shelf_col[type])

        # changed to three dimension
        batch = tgt_input.size()[1]
        src_pos_2d[-1, :, 2] = 1 # the last one is target
        for i in range(batch):
            Index = src_pos[-1, i]
            tgt1 = torch.where(tgt_input[:, i] == Index)[0]
            tgt_input_2d[tgt1, i, 2] = 1
        return src_pos_2d, tgt_input_2d

    def valid_one_dataset(self, batch, type):
        return self.train_one_dataset(batch, type, return_gaze=True)

    def validation_step(self, batch, batch_idx):
        data1, data2 = batch
        if len(data1)==0:
            #logits = self.train_one_dataset(data2, 1, True)
            loss, GAZE = self.valid_one_dataset(data2, 1)
            gt = data2[2][1:,:][:-1]
            GAZE = GAZE[0]
            target = data2[0][-1]
            #sim = saliency_map_metric(logits, data2[2][1:,0])
            ss = nw_matching(gt[:,0].detach().cpu().numpy(), GAZE[:,0].detach().cpu().numpy())
        else:
            #logits = self.train_one_dataset(data1, 0, True)
            loss, GAZE = self.valid_one_dataset(data1, 0)
            gt = data1[2][1:,:][:-1]
            GAZE = GAZE[0]
            target = data1[0][-1]
            #sim = saliency_map_metric(logits, data1[2][1:,0])
            ss = nw_matching(gt[:,0].detach().cpu().numpy(), GAZE[:,0].detach().cpu().numpy())
        sim=0
        self.log('validation_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'loss': loss, 'GAZE': GAZE,  'GAZE_gt': gt,  'target': target, 'sim': sim, 'ss': ss}

    def validation_epoch_end(self, validation_step_outputs):
        avg_loss = torch.stack([x['loss'] for x in validation_step_outputs]).mean()
        avg_sim = np.stack([x['sim'] for x in validation_step_outputs]).mean()
        avg_ss = np.stack([x['ss'] for x in validation_step_outputs]).mean()
        res_gt, res_max = torch.zeros(6).to(DEVICE), torch.zeros(6).to(DEVICE)
        i = 0
        for output in validation_step_outputs:
            gaze = output['GAZE'].cpu().detach().numpy().T
            gaze_gt = output['GAZE_gt'].cpu().detach().numpy().T
            target = output['target'].cpu().detach().numpy()
            behavior(res_gt, target, gaze_gt)
            behavior(res_max, target, gaze)
            i += 1
        res_gt = res_gt / i
        res_max = res_max / i
        res_max[5] = torch.mean(torch.abs(res_max[:5] - res_gt[:5]) / res_gt[:5])
        delta = res_max[5]
        #print('delta: ', delta)
        #print('sim: ', avg_sim)
        self.log('validation_loss_each_epoch', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('validation_delta_each_epoch', delta, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('validation_sim_each_epoch', avg_sim, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('validation_ss_each_epoch', avg_ss, on_epoch=True, prog_bar=True, sync_dist=True)


    def test_step(self, batch, batch_idx):
        data1, data2 = batch
        if len(data1) == 0: # 7, 1?
            loss_gt, GAZE = self.valid_one_dataset(data2, 1)
            GAZE_gt = data2[2][1:, :][:-1]
            GAZE = GAZE[0]
            ss = nw_matching(GAZE_gt[:, 0].detach().cpu().numpy(), GAZE[:, 0].detach().cpu().numpy())
        else:
            loss_gt, GAZE = self.valid_one_dataset(data1, 0)
            GAZE_gt = data1[2][1:, :][:-1]
            GAZE = GAZE[0]
            ss = nw_matching(GAZE_gt[:, 0].detach().cpu().numpy(), GAZE[:, 0].detach().cpu().numpy())

        self.log('testing_loss', loss_gt, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'loss_max': 0, 'loss_expect': 0, 'loss_gt': loss_gt,
                'GAZE': GAZE, 'GAZE_gt': GAZE_gt, 'GAZE_expect': 0, 'sim':0,
                'ss_max': ss, 'ss_exp': 0}

    def test_epoch_end(self, test_step_outputs):
        all_gaze, all_gaze_gt = pd.DataFrame(), pd.DataFrame()
        for output in test_step_outputs:
            gazes = output['GAZE'].cpu().detach().numpy().T
            all_gaze = pd.concat([all_gaze, pd.DataFrame(gazes)],axis=0)
            gazes_gt = output['GAZE_gt'].cpu().detach().numpy().T
            all_gaze_gt = pd.concat([all_gaze_gt, pd.DataFrame(gazes_gt)],axis=0)

        all_gaze.reset_index().drop(['index'],axis=1)
        all_gaze_gt.reset_index().drop(['index'],axis=1)
        all_gaze.to_csv(self.args.output_path + '/gaze_max' + self.args.output_postfix + '.csv', index=False)
        all_gaze_gt.to_csv(self.args.output_path + '/gaze_gt' + self.args.output_postfix + '.csv', index=False)

        avg_loss = torch.stack([x['loss_gt'].cpu().detach() for x in test_step_outputs]).mean()
        self.log('test_loss_gt_each_epoch', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)

        avg_ss_max = np.stack([x['ss_max'] for x in test_step_outputs]).mean()
        self.log('test_ss_max', avg_ss_max, on_epoch=True, prog_bar=True, sync_dist=True)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.scheduler_lambda1, gamma=self.args.scheduler_lambda2)
        return [optimizer], [scheduler]

