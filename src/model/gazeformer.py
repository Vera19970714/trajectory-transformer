import torch
import torch.nn.functional as F
from torch import nn, Tensor
import math
from .gazeformer_models import Transformer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PositionEmbeddingSine2d(nn.Module):
    def __init__(self, spatial_dim, hidden_dim=768, temperature=10000, normalize=False, scale=None, flatten = True, device = "cuda:0"):
        super(PositionEmbeddingSine2d, self).__init__()
        self.num_pos_feats = hidden_dim // 2
        normalize = normalize
        self.h, self.w = spatial_dim
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.device = device
        position_y = torch.arange(self.h).unsqueeze(1)
        position_x = torch.arange(self.w).unsqueeze(1)
        if normalize:
            eps = 1e-6
            position_y = position_y / (self.h - 1 + eps) * scale
            position_x = position_x / (self.w - 1 + eps) * scale
        div_term = torch.exp(torch.arange(0, self.num_pos_feats, 2) * (-math.log(temperature) / self.num_pos_feats))
        pe_y = torch.zeros(self.h, 1, self.num_pos_feats)
        pe_x = torch.zeros(1, self.w, self.num_pos_feats)
        pe_y[:, 0, 0::2] = torch.sin(position_y * div_term)
        pe_y[:, 0, 1::2] = torch.cos(position_y * div_term)
        pe_x[0, :, 0::2] = torch.sin(position_x * div_term)
        pe_x[0, :, 1::2] = torch.cos(position_x * div_term)
        pe_y = pe_y.repeat(1, self.w, 1)
        pe_x = pe_x.repeat(self.h, 1, 1)
        self.pos = torch.cat((pe_y, pe_x), dim=-1).permute(2, 0, 1)
        if flatten:
            self.pos =  self.pos.view(hidden_dim, -1).permute(1,0).unsqueeze(1)
        else:
            self.pos = self.pos.permute(1,2,0)
        del pe_y, pe_x, position_y, position_x

    def forward(self, x):
        return x.to(self.device) + self.pos.to(self.device)


class gazeformer(nn.Module):
    def __init__(self, spatial_dim, number_coder=4, number_heads=4, hidden_dim=512, dropout=0.4, max_len=7, patch_size=16):
        super(gazeformer, self).__init__()
        self.spatial_dim = spatial_dim
        self.transformer = Transformer(num_encoder_layers=number_coder, nhead=number_heads, d_model=hidden_dim,
                                  num_decoder_layers=number_coder, encoder_dropout=0.1,
                                  decoder_dropout=0.1, dim_feedforward=hidden_dim, device=device).to(device)

        self.hidden_dim = self.transformer.d_model
        # fixation embeddings
        self.querypos_embed = nn.Embedding(max_len, self.hidden_dim).to(device)
        # 2D patch positional encoding
        self.patchpos_embed = PositionEmbeddingSine2d(spatial_dim, hidden_dim=self.hidden_dim, normalize=True,
                                                      device=device)
        # 2D pixel positional encoding for initial fixation
        self.queryfix_embed = PositionEmbeddingSine2d((spatial_dim[0] * patch_size, spatial_dim[1] * patch_size),
                                                      hidden_dim=self.hidden_dim, normalize=True, flatten=False,
                                                      device=device).pos.to(device)
        # classify fixation, or PAD tokens
        self.token_predictor = nn.Linear(self.hidden_dim, 2).to(device)
        # Gaussian parameters for x,y,t
        self.generator_y_mu = nn.Linear(self.hidden_dim, 1).to(device)
        self.generator_x_mu = nn.Linear(self.hidden_dim, 1).to(device)
        #self.generator_t_mu = nn.Linear(self.hidden_dim, 1).to(device)
        self.generator_y_logvar = nn.Linear(self.hidden_dim, 1).to(device)
        self.generator_x_logvar = nn.Linear(self.hidden_dim, 1).to(device)
        #self.generator_t_logvar = nn.Linear(self.hidden_dim, 1).to(device)

        self.device = device
        self.max_len = max_len

        self.activation = F.relu
        self.dropout = nn.Dropout(dropout)

        self.softmax = nn.LogSoftmax(dim=-1).to(device)
        # projection for first fixation encoding
        self.firstfix_linear = nn.Linear(self.hidden_dim, self.hidden_dim)

    # reparameterization trick
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, tgt, src, task): # tgt is starting token
                #src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        #src = src.to(self.device)
        tgt_input = torch.zeros(self.max_len, src.size(0), self.hidden_dim).to(
            self.device)  # Notice that this where we convert target input to zeros
        tgt_input[0, :, :] = self.firstfix_linear(self.queryfix_embed[tgt[:, 0], tgt[:, 1], :])

        outs = self.transformer(src=src, tgt=tgt_input, src_mask=None, tgt_mask=None, memory_mask=None,
                                src_key_padding_mask=None, tgt_key_padding_mask=None,
                                memory_key_padding_mask=None,
                                task=task.to(self.device), querypos_embed=self.querypos_embed.weight.unsqueeze(1),
                                patchpos_embed=self.patchpos_embed)

        outs = self.dropout(outs)
        # get Gaussian parameters for (x,y,t)
        y_mu, y_logvar, x_mu, x_logvar = self.generator_y_mu(outs), self.generator_y_logvar(
            outs), self.generator_x_mu(outs), self.generator_x_logvar(outs) #, self.generator_t_mu(
            #outs), self.generator_t_logvar(outs)

        return self.softmax(self.token_predictor(outs)), self.activation(
            self.reparameterize(y_mu, y_logvar)), self.activation(self.reparameterize(x_mu, x_logvar)) #, self.activation(
            #self.reparameterize(t_mu, t_logvar))

