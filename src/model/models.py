from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class VisualPositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(VisualPositionalEncoding, self).__init__()
        pos_embedding = nn.Parameter(torch.randn(maxlen, emb_size))
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('visual_pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.visual_pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class CNNEmbedding(nn.Module):
    def __init__(self, outputSize):
        super(CNNEmbedding, self).__init__()
        self.cnn1 = nn.Sequential(nn.Conv2d(3, 16, (5, 5)), nn.ReLU(), nn.MaxPool2d(5))
        self.cnn2 = nn.Sequential(nn.Conv2d(16, 32, (3, 3)), nn.ReLU(), nn.MaxPool2d(3))
        self.fc = nn.Linear(1440, outputSize)
        #nn.init.kaiming_normal_(self.fc.weight, mode='fan_in',
        #                        nonlinearity='leaky_relu')

    def forward(self, tokens: Tensor):
        # 4, 28, 150, 93, 3
        b, l = tokens.size()[0], tokens.size()[1]
        w, h = tokens.size()[2], tokens.size()[3]
        input = tokens.contiguous().view(-1, w, h, 3).permute(0, 3, 1, 2)
        output = self.cnn1(input) # b, 16, 29, 17
        output = self.cnn2(output) # 112, 32, 9, 5
        output = torch.flatten(output, start_dim=1, end_dim=-1)
        output = self.fc(output)
        return output.view(b, l, -1)

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
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
        super(Seq2SeqTransformer, self).__init__()
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

    def encode(self, src: Tensor, src_mask: Tensor):
        return 0

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return 0


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt, PAD_IDX, isGlobalToken=False):
    src_seq_len = src.shape[0]
    if isGlobalToken:
        src_seq_len += 1
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    if isGlobalToken:
        token_pad = torch.tensor([[False]]).view(1, 1).to(DEVICE)
        token_pad = token_pad.repeat(src_padding_mask.size()[0], 1)
        src_padding_mask = torch.cat((token_pad, src_padding_mask), dim=1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


