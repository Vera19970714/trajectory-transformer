import numpy as np
import torch
import torch.nn as nn
import math
from numpy import dot
from numpy.linalg import norm
import matplotlib.pyplot as plt
from torch import Tensor

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, G: int, M: int, F_dim: int, H_dim: int, D: int, gamma: float):
        """
        Learnable Fourier Features from https://arxiv.org/pdf/2106.02795.pdf (Algorithm 1)
        Implementation of Algorithm 1: Compute the Fourier feature positional encoding of a multi-dimensional position
        Computes the positional encoding of a tensor of shape [N, G, M]
        :param G: positional groups (positions in different groups are independent)
        :param M: each point has a M-dimensional positional values
        :param F_dim: depth of the Fourier feature dimension
        :param H_dim: hidden layer dimension
        :param D: positional encoding dimension
        :param gamma: parameter to initialize Wr
        """
        super().__init__()
        self.G = G
        self.M = M
        self.F_dim = F_dim
        self.H_dim = H_dim
        self.D = D
        self.gamma = gamma

        # Projection matrix on learned lines (used in eq. 2)
        self.Wr = nn.Linear(self.M, self.F_dim // 2, bias=False)
        # MLP (GeLU(F @ W1 + B1) @ W2 + B2 (eq. 6)
        self.mlp = nn.Sequential(
            nn.Linear(self.F_dim, self.H_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.H_dim, self.D // self.G)
        )

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma ** -2)

    def forward(self, x):
        """
        Produce positional encodings from x
        :param x: tensor of shape [N, G, M] that represents N positions where each position is in the shape of [G, M],
                  where G is the positional group and each group has M-dimensional positional values.
                  Positions in different positional groups are independent
        :return: positional encoding for X
        """
        B, N, G, M = x.shape
        # Step 1. Compute Fourier features (eq. 2)
        projected = self.Wr(x)
        cosines = torch.cos(projected)
        sines = torch.sin(projected)
        F = 1 / np.sqrt(self.F_dim) * torch.cat([cosines, sines], dim=-1)
        # Step 2. Compute projected Fourier features (eq. 6)
        Y = self.mlp(F)
        # Step 3. Reshape to x's shape
        PEx = Y.reshape((B, N, self.D))
        return PEx


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels, N=10000):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (N ** (torch.arange(0, channels, 2).float() / channels))
        #inv_freq = (torch.arange(0, channels, 2).float()) ** 1 / channels
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1)
        emb_y = get_emb(sin_inp_y)
        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        self.cached_penc = emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)
        return self.cached_penc


class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)

    @property
    def org_channels(self):
        return self.penc.org_channels


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels, changeX, functionChoice, alpha):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding3D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        if functionChoice == 'original':
            inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        elif functionChoice == 'exp1':
            inv_freq = (torch.arange(0, channels, 2).float()) ** alpha / channels
        elif functionChoice == 'exp2':
            inv_freq = (torch.arange(0, channels, 2).float() / channels) ** alpha
        elif functionChoice == 'linear':
            inv_freq = (torch.arange(0, channels, 2).float()) * alpha / channels

        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None)
        self.changeX = changeX

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        if self.changeX == 'True':
            pos_x = torch.arange(0, 4, 1.6, device=tensor.device).type(self.inv_freq.type())
        elif self.changeX == 'False':
            pos_y = torch.arange(0, 5, 0.6, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1).unsqueeze(1)
        emb_y = get_emb(sin_inp_y).unsqueeze(1)
        emb_z = get_emb(sin_inp_z)
        emb = torch.zeros((x, y, z, self.channels * 3), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels : 2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels :] = emb_z

        self.cached_penc = emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)
        return self.cached_penc


class PositionalEncodingPermute3D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y, z) instead of (batchsize, x, y, z, ch)
        """
        super(PositionalEncodingPermute3D, self).__init__()
        self.penc = PositionalEncoding3D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 4, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 4, 1, 2, 3)

    @property
    def org_channels(self):
        return self.penc.org_channels


def getFourierPositional(dimension, embed):
    G = 1
    M = dimension
    F = D = embed
    Gamma = 10
    H = 32
    enc = LearnableFourierPositionalEncoding(G, M, F, H, D, Gamma)
    return enc

def getSinPositional(dimension, embed, functionChoice, alpha, changeX):
    if dimension == 2:
        enc = PositionalEncoding2D(embed)
        x = enc(torch.randn(1, 3, 9, embed))
    elif dimension == 3:
        enc = PositionalEncoding3D(embed, changeX, functionChoice, alpha)
        x = enc(torch.randn(1, 3, 9, 2, embed))
    return x

def calculate2DPositional(x, src):
    #x: 1, 3, 9, 256; src: 28, 1, 3
    tgt = x[:, src[0, :, 0].long(), src[0, :, 1].long(), :] # 1, 256
    shelf = x[0].view(27, -1).unsqueeze(1).repeat(1, tgt.size()[1], 1)
    pos = torch.cat((tgt, shelf), dim=0)
    return pos


def calculate3DPositional(x, src):
    #x: 1, 3, 9, 2, 256; src: 28, 1, 3
    '''tgt = x[:, src[0, :, 0].long(), src[0, :, 1].long(), 1, :] # 1, 256
    shelf = x[0, :, :, 0, :].view(27, -1).unsqueeze(1).repeat(1, tgt.size()[1], 1)
    pos = torch.cat((tgt, shelf), dim=0)
    return pos'''
    res = torch.zeros((src.size()[0], src.size()[1], x.size()[-1])).to(DEVICE)
    for i in range(src.size()[0]):
        for j in range(src.size()[1]):
            a = src[i, j].long()
            try:
                res[i, j] = x[0, a[0], a[1], a[2]]
            except:
                pass
    return res


def getCosSim(a, b):
    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return cos_sim


class PositionalEncoding2DUpdated(nn.Module):
    def __init__(self, channels, changeX):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        changeX: True or False, change x index or y. None: do nothing
        """
        super(PositionalEncoding2DUpdated, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        #inv_freq = (torch.arange(0, channels, 2).float())**(0.8) / channels
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None)
        self.changeX = changeX

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type()) # 50,
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type()) # 50,
        if self.changeX is True:
            pos_x = torch.arange(0, 4, 1.6, device=tensor.device).type(self.inv_freq.type())
        elif self.changeX is False:
            pos_y = torch.arange(0, 5, 0.6, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        #heat_map = sns.heatmap(sin_inp_x, linewidth=1, annot=False)
        #plt.show()

        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1)
        emb_y = get_emb(sin_inp_y)
        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        self.cached_penc = emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)
        return self.cached_penc


def getFakeFourier(ind, W):
    # ind: 1, 2, W: 2, 50
    f = np.matmul(ind, W) # 1, 50
    emb = np.concatenate([np.sin(f), np.cos(f)])  # 1, 100
    return emb

def computeDotProductDistribution(p1, p2):
    # p1 size: N, D; p2 size: N, D, 1D positional
    # run decoder 1) p1=img concat 3D, 2) p1=img add 3D, 3) p1=(img add 3D)*projection 4) p1=img 5) p1=3D
    sims = []
    assert p1.size() == p2.size()
    N, D = p1.size()
    for i in range(N):
        for j in range(N):
            a = getCosSim(p1[i].detach().cpu().numpy(), p2[j].detach().cpu().numpy())
            sims.append(a)

    plt.hist(sims, bins=5)
    plt.show()


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 functionChoice: str,
                 alpha: int,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()

        #den = (torch.arange(0, emb_size, 2).float()) ** 0.8 / emb_size # exp1
        #den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size) #original
        channels = emb_size
        if functionChoice == 'original':
            inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        elif functionChoice == 'exp1':
            inv_freq = (torch.arange(0, channels, 2).float()) ** alpha / channels
        elif functionChoice == 'exp2':
            inv_freq = (torch.arange(0, channels, 2).float() / channels) ** alpha
        elif functionChoice == 'linear':
            inv_freq = (torch.arange(0, channels, 2).float()) * alpha / channels
        den = inv_freq

        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


def draw2Dheatmap():
    import seaborn as sns
    import matplotlib.pylab as plt

    embed = 256
    totali = 50
    totalj = 50
    enc = PositionalEncoding2DUpdated(embed, changeX=None)
    x = enc(torch.randn(1, totali, totalj, embed)).numpy()[0]  # 50, 50, 100
    center = x[int(totali / 2), int(totalj / 2)]
    simMatrix = np.zeros((totali, totalj))
    # W = np.random.normal(0, 1, size=(2, 50))
    for i in range(totali):
        for j in range(totalj):
            emb = x[i][j]
            # emb = getFakeFourier(np.array([i, j]), W)
            a = getCosSim(emb, center)
            simMatrix[i][j] = a

    heat_map = sns.heatmap(simMatrix, linewidth=1, annot=False)
    plt.show()


def draw1Dheatmap():
    import seaborn as sns
    import matplotlib.pylab as plt

    embed = 256
    totali = 15
    totalj = 50
    def getLine(choice, alpha):
        enc = PositionalEncoding(embed, 0, choice, alpha).pos_embedding
        x = enc[:totali, 0].numpy()  # 50, 50, 100
        ref = x[int(totali/2)]
        simMatrix = np.zeros((1, totali))
        # W = np.random.normal(0, 1, size=(2, 50))
        for i in range(1, totali):
            emb = x[i]
            a = getCosSim(emb, ref)
            # a = getCosSim(emb, x[i-1])
            simMatrix[0, i] = a
        return simMatrix[0, 1:]

    line9 = getLine('original', 0.9)
    line8 = getLine('exp1', 0.9)
    line7 = getLine('exp1', 0.7)
    line6 = getLine('exp1', 0.5)
    line5 = getLine('linear', 0.9)
    line4 = getLine('exp2', 0.9)
    #sns.heatmap(simMatrix, linewidth=1, annot=False)
    #print(simMatrix[0])
    #sns.heatmap(simMatrix[:, 1:], linewidth=1, annot=False)
    # plt.hist(simMatrix[0, 1:])
    plt.plot(line4, label='exp2, 0.9')
    plt.plot(line5, label='linear, 0.9')
    plt.plot(line6, label='exp1, 0.5')
    plt.plot(line7, label='exp1, 0.7')
    plt.plot(line8, label='exp1, 0.9')
    plt.plot(line9, label='original')
    plt.legend()
    plt.show()


def runDifferentDecoderPE():
    # run decoder 1) p1=img concat 3D, 2) p1=img add 3D, 3) p1=(img add 3D)*projection 4) p1=img 5) p1=3D

    lw = torch.from_numpy(np.load('dataset/PE/linearWeight.npy'))  # 256,256
    tgt_cnn_emb = torch.from_numpy(np.load('dataset/PE/tgt_cnn_emb.npy'))  # 8,1,256
    tg = torch.from_numpy(np.load('dataset/PE/trg.npy'))  # 8,1,3
    emb_size = 512
    threedSin = getSinPositional(3, int(emb_size / 2), 'original', 0.8, None)
    tgt_pos_emb = calculate3DPositional(threedSin, tg).to(DEVICE)
    onedpositional_encoding = PositionalEncoding(
        int(emb_size / 2), dropout=0.1)
    p1 = (tgt_cnn_emb + tgt_pos_emb)  # torch.cat((tgt_cnn_emb, tgt_pos_emb), dim=2)
    p1 = torch.matmul(p1, lw)
    p2 = onedpositional_encoding(p1)

    computeDotProductDistribution(p1.squeeze(1), p2.squeeze(1))

if __name__ == '__main__':
    '''G = 3
    M = 17
    x = torch.randn((97, G, M))
    enc = LearnableFourierPositionalEncoding(G, M, 768, 32, 768, 10)
    pex = enc(x)
    print(pex.shape)'''
    draw1Dheatmap()


