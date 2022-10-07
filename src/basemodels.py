from torch import Tensor
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import math
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 27, 28, 29, 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class CNNShelf(nn.Module):
    def __init__(self):
        super(CNNShelf, self).__init__()
        self.cnn1 = nn.Sequential(nn.Conv2d(3, 8, (5, 5)), nn.ReLU(), nn.MaxPool2d(5))
        self.cnn2 = nn.Sequential(nn.Conv2d(8, 16, (3, 3)), nn.ReLU(), nn.MaxPool2d(3))
        self.cnn3 = nn.Sequential(nn.Conv2d(16, 32, (3, 3)), nn.ReLU(), nn.MaxPool2d(3))
        self.fc1 = nn.Linear(4896, 1024)
        

    def forward(self, tokens: Tensor):
        # b，837, 450, 3
        b, w, h = tokens.size()[0], tokens.size()[1], tokens.size()[2]
        input = tokens.contiguous().view(-1, w, h, 3).permute(0, 3, 1, 2)
        print('=====input===')
        print(input.shape)
        output = self.cnn1(input) # b, 8, 166，89
        output = self.cnn2(output) # b, 16, 54, 29
        output = self.cnn3(output) # b, 32, 17, 9
        output = torch.flatten(output, start_dim=1, end_dim=-1)
        output = self.fc1(output)
        return output.view(b,-1)

class CNNEmbedding(nn.Module):
    def __init__(self):
        super(CNNEmbedding, self).__init__()
        self.cnn1 = nn.Sequential(nn.Conv2d(3, 16, (5, 5)), nn.ReLU(), nn.MaxPool2d(5))
        self.cnn2 = nn.Sequential(nn.Conv2d(16, 32, (3, 3)), nn.ReLU(), nn.MaxPool2d(3))
        self.fc = nn.Linear(1440, 256)

    def forward(self, tokens: Tensor):
        # b, 28, 150, 93, 3
        b, l = tokens.size()[0], tokens.size()[1]
        w, h = tokens.size()[2], tokens.size()[3]
        input = tokens.contiguous().view(-1, w, h, 3).permute(0, 3, 1, 2)
        output = self.cnn1(input) # 112, 16, 29, 17
        output = self.cnn2(output) # 112, 32, 9, 5
        output = torch.flatten(output, start_dim=1, end_dim=-1)
        output = self.fc(output)
        return output.view(b, l, -1)

# Seq2Seq Network 

class DecoderRNN(nn.Module):
    def __init__(self, 
                embed_size: int, 
                hidden_size: int,
                feature_size: int, 
                vocab_size: int, 
                num_layers: int):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = TokenEmbedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.generator = nn.Linear(hidden_size, vocab_size)
        self.cnn_embedding = CNNEmbedding()
        self.cnn_shelf = CNNShelf()
        self.featurecomb = nn.Linear(feature_size,  embed_size)
        self.tgt_tok_emb = TokenEmbedding(vocab_size, embed_size)
        
    def forward(self,
                trg: Tensor,
                src_img: Tensor,
                question_img: Tensor,
                tgt_img: Tensor):
        """Decode image feature vectors and generates gazes."""
        # print('====inut size=====')
        # print(trg.shape)
        # print(src_img.shape)
        # print(question_img.shape)
        # print(tgt_img.shape)

        src_cnn_emb = self.cnn_embedding(src_img).squeeze(1) #b,256

        # print('=====cnnemb size======')
        # print(src_cnn_emb.shape)
        question_cnn_emb = self.cnn_shelf(question_img)#b,1024
        # print('=====shelfemb size======')
        # print(question_cnn_emb.shape)
        src_emb = torch.cat((question_cnn_emb,src_cnn_emb), dim=1) #b, 1280
        # print('=====emb size======')
        # print(src_emb.shape)
        src_emb = self.featurecomb(src_emb).unsqueeze(1) #b,256
        
        # tgt_cnn_emb = self.cnn_embedding(tgt_img).transpose(0, 1)  # 28, 4, 256
        tgt_emb = self.tgt_tok_emb(trg).transpose(0, 1)  # b, 28, 256
        # print('=====tgt size======')
        # print(tgt_emb.shape)
        embeddings = torch.cat((src_emb, tgt_emb), 1)
        # print('=====embeddings size======')
        # print(embeddings.shape)
        hiddens, _ = self.lstm(embeddings)
        # print('=====hidden size======')
        # print(hiddens.shape)
        outputs = self.generator(hiddens)
        # print('=====out size======')
        # print(outputs.shape)
        # exit()
        return outputs
        

    
    # def sample(self, features, states=None):
    #     """Generate captions for given image features using greedy search."""
    #     sampled_ids = []
    #     inputs = features.unsqueeze(1)
    #     for i in range(self.max_seg_length):
    #         hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
    #         outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
    #         _, predicted = outputs.max(1)                        # predicted: (batch_size)
    #         sampled_ids.append(predicted)
    #         inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
    #         inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
    #     sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
    #     return sampled_ids

