from torch.nn.modules.activation import ReLU
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as torch_models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import pandas as pd

class Conv_Autoencoder(nn.Module):
    def __init__(self, args):
        super(Conv_Autoencoder, self).__init__()
        self.args = args

        self.normalize = transforms.Normalize((0.5), (0.5))

        # self.encoder_conv1 = nn.Conv2d(3, 16, (3, 3), stride=(3, 3), padding=(3, 0))  # b, 16, 102, 62
        # self.encoder_maxpool1 = nn.MaxPool2d(2, stride=2, return_indices=True) # b, 16, 51, 31
        # self.encoder_conv2 = nn.Conv2d(16, 32, (3, 3), stride=(2, 2), padding=(1, 1))  # b, 32, 26, 16
        # self.encoder_maxpool2 = nn.MaxPool2d(2, stride=1, return_indices=True) # b, 16, 25, 15
        # self.encoder_conv3 = nn.Conv2d(32, 64, (5, 5), stride=(2, 2), padding=(0, 0))  # b, 64, 11, 6
        # self.encoder_maxpool3 = nn.MaxPool2d(2, stride=1, return_indices=True)  # b, 64, 10, 5
        # self.encoder_conv4 = nn.Conv2d(64, 128, (3, 3), stride=(3, 2), padding=(1, 1))  # b, 128, 4, 3
        # self.encoder_maxpool4 = nn.MaxPool2d((2,2), stride=(2,1), return_indices=True)  # b, 128, 2, 2



        # self.decoder_maxunpool1 = nn.MaxUnpool2d((2,2), stride=(2,1)) # b, 128, 3, 4
        # self.decoder_unconv1 = nn.ConvTranspose2d(128, 64, (3, 3), stride=(3, 2), padding=(1, 1))  # b, 64, 10, 5
        # self.decoder_maxunpool2 = nn.MaxUnpool2d(2, stride=1) # b, 64, 11, 6
        # self.decoder_unconv2 = nn.ConvTranspose2d(64, 32, (5, 5), stride=(2, 2), padding=(0, 0))  # b, 32, 25, 15
        # self.decoder_maxunpool3 = nn.MaxUnpool2d(2, stride=1) # b, 32, 26, 16
        # self.decoder_unconv3 = nn.ConvTranspose2d(32, 16, (3, 3), stride=(2, 2), padding=(1, 1))  # b, 16, 51, 31
        # self.decoder_maxunpool4 = nn.MaxUnpool2d(2, stride=2) # b, 16, 102, 62
        # self.decoder_unconv4 = nn.ConvTranspose2d(16, 3, (3, 3), stride=(3, 3), padding=(3, 0))  # b, 3, 300, 186
        # self.activate = nn.ReLU()
        # self.output = nn.Sigmoid()

        self.encoder_conv1 = nn.Conv2d(3, 16, (3, 3), stride=(3, 3), padding=(0, 0))  # b, 16, 100, 62
        self.encoder_conv2 = nn.Conv2d(16, 32, (3, 3), stride=(1, 1), padding=(1, 0))  # b, 32, 100, 60
        self.encoder_conv3 = nn.Conv2d(32, 64, (3, 3), stride=(3, 3), padding=(1, 0))  # b, 64, 34, 20
        self.encoder_conv4 = nn.Conv2d(64, 128, (3, 3), stride=(3, 3), padding=(1, 2))  # b, 128, 12, 8
        # self.encoder_conv5 = nn.Conv2d(128, 256, (3, 3), stride=(3, 3), padding=(2, 1))  # b, 128, 12, 8


        self.decoder_unconv1 = nn.ConvTranspose2d(128, 64, (3, 3), stride=(3, 3), padding=(1, 2))  # b, 64, 34, 20
        self.decoder_unconv2 = nn.ConvTranspose2d(64, 32, (3, 3), stride=(3, 3), padding=(1, 0))  # b, 32, 100, 60
        self.decoder_unconv3 = nn.ConvTranspose2d(32, 16, (3, 3), stride=(1, 1), padding=(1, 0))  # b, 16, 100, 62
        self.decoder_unconv4 = nn.ConvTranspose2d(16, 3, (3, 3), stride=(3, 3), padding=(0, 0))  # b, 3, 300, 186
        self.activate = nn.ReLU()
        self.output = nn.Sigmoid()





        self.loss = nn.MSELoss()

    def forward(self, input_img):
        # encoder


        input_img = torch.squeeze(input_img, 1)
        # print(input_img.shape)
        # print(input_img.shape)
        hidden_img= self.encoder_conv1(input_img)

        hidden_img =self.activate(hidden_img)
        # hidden_img, indices1 = self.encoder_maxpool1(hidden_img)
        hidden_img = self.encoder_conv2(hidden_img)
        hidden_img = self.activate(hidden_img)
        # hidden_img, indices2 = self.encoder_maxpool2(hidden_img)
        hidden_img = self.encoder_conv3(hidden_img)
        hidden_img = self.activate(hidden_img)
        # hidden_img, indices3 = self.encoder_maxpool3(hidden_img)
        hidden_img = self.encoder_conv4(hidden_img)

        # hidden_img = self.activate(hidden_img)
        # print(hidden_img.shape)
        # exit()
        # hidden_img, indices4 = self.encoder_maxpool4(hidden_img)

        #decoder
        # out_img = self.decoder_maxunpool1(hidden_img, indices4)
        # out_img = self.activate(out_img)
        out_img = self.decoder_unconv1(hidden_img)

        # out_img = self.decoder_maxunpool2(out_img, indices3)
        out_img = self.activate(out_img)
        out_img = self.decoder_unconv2(out_img)

        # out_img = self.decoder_maxunpool3(out_img, indices2)
        out_img = self.activate(out_img)
        out_img = self.decoder_unconv3(out_img)
        # out_img = self.decoder_maxunpool4(out_img, indices1)
        out_img = self.activate(out_img)

        out_img = self.decoder_unconv4(out_img)



        out_img = self.output(out_img)


        loss = self.loss(out_img, input_img)
        return loss, input_img, hidden_img, out_img


class Conv_AutoencoderModel(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = Conv_Autoencoder(self.args)
        
    def training_step(self, batch, batch_idx):
        # batch
        img_feature = batch
        # get loss
        loss, input_img, hidden_img, out_img = self.model(img_feature)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'loss': loss, 'input_img':input_img, 'hidden_img':hidden_img, 'out_img':out_img}

    def training_epoch_end(self, training_step_outputs):
        avg_loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
        self.log('training_loss_each_epoch', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)


    def validation_step(self, batch, batch_idx):
        # batch
        img_feature = batch
        # get loss
        loss, input_img, hidden_img, out_img = self.model(img_feature)
        self.log('validation_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'loss': loss, 'input_img':input_img, 'hidden_img':hidden_img, 'out_img':out_img}

    def validation_epoch_end(self, validation_step_outputs):
        avg_loss = torch.stack([x['loss'] for x in validation_step_outputs]).mean()
        self.log('validation_loss_each_epoch', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)


    def test_step(self, batch, batch_idx):
        # batch
        img_feature = batch
        # get loss
        loss, input_img, hidden_img, out_img = self.model(img_feature)
        self.log('validation_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'loss': loss, 'input_img':input_img.cpu().detach(), 'hidden_img':hidden_img.cpu().detach(), 'out_img':out_img.cpu().detach()}

    def test_epoch_end(self, test_step_outputs):
        avg_loss = torch.stack([x['loss'].cpu().detach() for x in test_step_outputs]).mean()
        # input_img_feature = torch.cat([x['input_img'].cpu().detach() for x in test_step_outputs])
        hidden_img_feature = torch.cat([x['hidden_img'].cpu().detach() for x in test_step_outputs])
        # output_img_feature = torch.cat([x['out_img'].cpu().detach() for x in test_step_outputs])
        iter_num = hidden_img_feature.shape[0]
        
        # # save image
        # output_image_save_path = './data/test_img_1/output_img/'
        # input_image_save_path = './data/test_img_1/input_img/'

        # for i in range(iter_num):
        #     input_img = input_img_feature[i].permute(1,2,0).cpu().detach().numpy()
        #     output_img = output_img_feature[i].permute(1,2,0).cpu().detach().numpy()
        #     input_img = Image.fromarray((input_img*255).astype(np.uint8)).convert('RGB')
        #     input_img.save(input_image_save_path + 'input' + '_' + str(i) + '.jpg')
        #     output_img = Image.fromarray((output_img*255).astype(np.uint8)).convert('RGB')
        #     output_img.save(output_image_save_path + 'onput' + '_' + str(i) + '.jpg')

        # reconstruct to whole question img
        hidden_img_dataset = []
        question_img_feature = []
        for i in range(iter_num):
            question_img_feature.append(hidden_img_feature[i])
            
            if (i + 1) % 27 == 0 and i > 0:
                dataset_dict = {}
                print(len(question_img_feature))
                dataset_dict['question_img_feature'] =  question_img_feature
                question_img_feature = []
                hidden_img_dataset.append(dataset_dict)
        print('='*10)
        print(len(hidden_img_dataset))

        torch.save(hidden_img_dataset, "./data/CAE_test/question_no_Q23")
        print('Finish saving...')
        self.log('test_loss_each_epoch', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.scheduler_lambda1, gamma=self.args.scheduler_lambda2)
        return [optimizer], [scheduler]

