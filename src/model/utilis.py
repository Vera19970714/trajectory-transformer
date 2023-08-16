import numpy as np
import torch
import torch.nn.functional as F
import cv2

class AttentionPlot():
    def __init__(self):
        pass
    def get_attention_map(self,img,encoder_atten,decoder_atten, isVit=False):
        if isVit:
            encoder_joint_atten = self.get_joint_attention(encoder_atten)
            ev = encoder_joint_atten[-1]
        else:
            ev = encoder_atten[-1,:,:].squeeze(0)
        # encoder_grid_size = int(np.sqrt(encoder_atten.size(-1)))
        # print(encoder_atten.size())
        # print(encoder_grid_size)
        
        encoder_mask = ev[0, 1:].reshape(3, 9).cpu().detach().numpy()
        encoder_mask = cv2.normalize(encoder_mask, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        encoder_mask = cv2.resize(encoder_mask, img.size)[..., np.newaxis]
        encoder_result = (encoder_mask * img).astype("uint8")

        decoder_result_all = []
        dv = decoder_atten[-1,:,:].squeeze(0)
        # decoder_grid_size = int(np.sqrt(decoder_atten.size(-1)))
        for i in range(dv.size(0)):
            decoder_mask = dv[i, 1:].reshape(3, 9).cpu().detach().numpy()
            decoder_mask = cv2.normalize(decoder_mask, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            decoder_mask = cv2.resize(decoder_mask, img.size)[..., np.newaxis]
            decoder_result = (decoder_mask * img).astype("uint8")
            decoder_result_all.append(decoder_result)
        return encoder_result, decoder_result_all
    
    def get_joint_attention(self,attention):
        # To account for residual connections, add identity matrix to the attention matrix and re-normalize the weights
        residual_att = torch.eye(attention.size(1))
        aug_att_mat = attention + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
        # Recursively multiply the weight matrices
        joint_attentions = torch.zeros(aug_att_mat.size())
        joint_attentions[0] = aug_att_mat[0]
        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])
        return joint_attentions, aug_att_mat.size(-1)