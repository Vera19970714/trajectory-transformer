from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
import torch.nn.functional as F
from transformerLightning import PositionalEncoding, VisualPositionalEncoding, TokenEmbedding
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SPPLayer(nn.Module):
    def __init__(self, num_levels=4, pool_type='max'):
        super(SPPLayer, self).__init__()
        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        bs, c, h, w = x.size()
        pooling_layers = []
        previous_conv_size = [h, w]
        for i in range(self.num_levels):
            level = 2 ** i
            h_kernel = int(math.ceil(previous_conv_size[0] / level))
            w_kernel = int(math.ceil(previous_conv_size[1] / level))
            w_pad1 = int(math.floor((w_kernel * level - previous_conv_size[1]) / 2))
            w_pad2 = int(math.ceil((w_kernel * level - previous_conv_size[1]) / 2))
            h_pad1 = int(math.floor((h_kernel * level - previous_conv_size[0]) / 2))
            h_pad2 = int(math.ceil((h_kernel * level - previous_conv_size[0]) / 2))
            assert w_pad1 + w_pad2 == (w_kernel * level - previous_conv_size[1]) and \
                   h_pad1 + h_pad2 == (h_kernel * level - previous_conv_size[0])

            padded_input = F.pad(input=x, pad=[w_pad1, w_pad2, h_pad1, h_pad2],
                                 mode='constant', value=0)
            if self.pool_type == "max":
                pool = nn.MaxPool2d((h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0))
            elif self.pool_type == "avg":
                pool = nn.AvgPool2d((h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0))
            else:
                raise RuntimeError("Unknown pooling type: %s, please use \"max\" or \"avg\".")

            tensor = pool(padded_input)
            tensor = torch.flatten(tensor, start_dim=1, end_dim=-1)
            pooling_layers.append(tensor)
        x = torch.cat(pooling_layers, dim=-1)
        return x

class CNNEmbedding(nn.Module):
    def __init__(self, outputSize):
        super(CNNEmbedding, self).__init__()
        self.cnn1 = nn.Sequential(nn.Conv2d(3, 16, (5, 5)), nn.ReLU(), nn.MaxPool2d(5))
        #self.cnn2 = nn.Sequential(nn.Conv2d(16, 32, (3, 3)), nn.ReLU(), nn.MaxPool2d(3))
        self.cnn2 = nn.Sequential(nn.Conv2d(16, 32, (3, 3)), nn.ReLU())
        self.fc = nn.Linear(1360, outputSize)
        self.sppLayer = SPPLayer()
        # nn.init.kaiming_normal_(self.fc.weight, mode='fan_in',
        #                        nonlinearity='leaky_relu')

    def forward(self, x: Tensor):
        b, l = len(x), x[0].size()[0]
        #w, h = x[0].size()[1], x[0].size()[2]
        outputs = []
        for i in range(b):
            tokens = x[i]
            input = tokens.permute(0, 3, 1, 2)
            output = self.cnn1(input)
            # SPP
            output = self.sppLayer(output)
            outputs.append(output)
        outputs = torch.stack(outputs)
        outputs = self.fc(outputs)
        return outputs.view(b, l, -1)

# Seq2Seq Network
class Seq2SeqTransformer4MIT1003(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 input_dimension: int,
                 dim_feedforward: int,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer4MIT1003, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size).float()
        #self.src_tok_emb = TokenEmbedding(src_vocab_size, int(emb_size/2))
        #self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, int(emb_size/2))
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)
        self.visual_positional_encoding = VisualPositionalEncoding(emb_size, dropout=dropout)

        self.cnn_embedding = CNNEmbedding(int(emb_size/2))
        self.LinearEmbedding = nn.Linear(input_dimension, int(emb_size/2))


    def getCNNFeature(self, src_img: Tensor):
        with torch.no_grad():
            src_cnn_emb = self.cnn_embedding(src_img).transpose(0, 1)
        return src_cnn_emb

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_img: Tensor,
                tgt_img: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_cnn_emb = self.cnn_embedding(src_img).transpose(0, 1) #28, 4, 256
        #src_pos_emb = self.src_tok_emb(src) # 28, 4, 256
        src_pos_emb = self.LinearEmbedding(src)
        src_emb = torch.cat((src_cnn_emb, src_pos_emb), dim=2) #28, 1, 384(256+128)
        src_emb = self.visual_positional_encoding(src_emb) #28,4,512
        #src_emb = self.positional_encoding(src_emb) #CHANGE: use positional encoding as well

        tgt_cnn_emb = self.cnn_embedding(tgt_img).transpose(0, 1)  # 28, 4, 256
        #tgt_pos_emb = self.tgt_tok_emb(trg)  # 28, 4, 256
        tgt_pos_emb = self.LinearEmbedding(trg)
        tgt_emb = torch.cat((tgt_cnn_emb, tgt_pos_emb), dim=2)
        tgt_emb = self.positional_encoding(tgt_emb)

        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

if __name__ == '__main__':
    cnn = CNNEmbedding(512)
    input = []
    input.append(torch.randn((16, 96, 128, 3)))
    #input.append(torch.randn((16, 128, 96, 3)))
    input.append(torch.randn((16, 72, 128, 3)))
    output = cnn(input)
    a = 1