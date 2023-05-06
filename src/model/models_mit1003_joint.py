from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
import torch.nn.functional as F
from .transformerLightning import PositionalEncoding, VisualPositionalEncoding, TokenEmbedding
from .models_mit1003 import *
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Seq2Seq Network
class Seq2SeqTransformer4MIT1003_Joint(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 input_dimension: int,
                 dim_feedforward: int,
                 isCNNExtractor=True, isDecoderOutputFea=True, isGlobalToken=True,
                 add_salient_OD=False, architecture_mode='scanpath',
                 dropout: float = 0.1):
        super(Seq2SeqTransformer4MIT1003_Joint, self).__init__()

        self.isDecoderOutputFea = isDecoderOutputFea
        self.isGlobalToken = isGlobalToken
        # shared part:
        self.cnn_embedding = CNNEmbedding(int(emb_size / 2), isCNNExtractor, add_salient_OD).float()
        self.visual_positional_encoding = VisualPositionalEncoding(emb_size, dropout=dropout)
        self.LinearEmbedding = nn.Linear(input_dimension, int(emb_size / 2))
        if self.isGlobalToken:
            self.globalToken = nn.Parameter(torch.randn(1, 1, emb_size))

        # heatmap generator:

        # scanpath generator:
        # todo: divide into two parts
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)

        if not self.isDecoderOutputFea:
            self.LinearEmbedding_decoder = nn.Linear(input_dimension, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size).float()


    def forward(self, src: Tensor, trg: Tensor,
                src_img: Tensor, tgt_img: Tensor, src_mask: Tensor, tgt_mask: Tensor, memory_key_padding_mask: Tensor,
                heatmaps: Tensor):
        src_cnn_emb = self.cnn_embedding(src_img).transpose(0, 1) #28, 4, 256
        #src_pos_emb = self.src_tok_emb(src) # 28, 4, 256

        src_pos_emb = self.LinearEmbedding(src)
        src_emb = torch.cat((src_cnn_emb, src_pos_emb), dim=2) #28, 1, 384(256+128)
        #src_emb = src_cnn_emb

        if self.isGlobalToken:
            bs = src_emb.size()[1]
            cls_tokens = self.globalToken.repeat(1, bs, 1) # 1,2,512
            src_emb = torch.cat((cls_tokens, src_emb), dim=0)
        src_emb = self.visual_positional_encoding(src_emb) #28,4,512
        if self.isDecoderOutputFea:
            tgt_cnn_emb = self.cnn_embedding(tgt_img).transpose(0, 1)  # 28, 4, 256
            tgt_pos_emb = self.LinearEmbedding(trg)
            tgt_emb = torch.cat((tgt_cnn_emb, tgt_pos_emb), dim=2)
        else:
            tgt_emb = self.LinearEmbedding_decoder(trg)
        tgt_emb = self.positional_encoding(tgt_emb)

        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

