import numpy as np
from torch.utils.data import DataLoader
from data_builder import FixDataset, collate_fn
import torch
# generate random integer values
from random import seed
from random import randint
# seed random number generator
#seed(1)

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 27, 28, 29, 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_datapath = '../dataset/processdata/dataset_Q23_time_val'
test_set = FixDataset(0, test_datapath)
test_loader = DataLoader(dataset=test_set, batch_size=1, num_workers=0, collate_fn=collate_fn, shuffle=False)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
losses = 0
time = 0
for src_pos, src_img, tgt_pos, tgt_img in test_loader:
    time += 1
    src_pos = src_pos.to(DEVICE)
    src_img = src_img.to(DEVICE)
    tgt_pos = tgt_pos.to(DEVICE)
    tgt_img = tgt_img.to(DEVICE)

    tgt_input = tgt_pos[:-1, :]
    tgt_img = tgt_img[:, :-1, :, :, :]

    tgt_out = tgt_pos[1:, :]

    output = torch.zeros((tgt_out.size()[0], tgt_out.size()[1], 31))
    for i in range(tgt_out.size()[0]):
        for j in range(tgt_out.size()[1]):
            ind = int(randint(0, 30))
            output[i][j][ind] = 1
    loss = loss_fn(output.reshape(-1, output.shape[-1]), tgt_out.reshape(-1))

    losses += loss.item()
print(losses / time)
