import numpy as np
from torch.utils.data import DataLoader
from data_builder import FixDataset, collate_fn
import torch
# generate random integer values
from random import seed
from random import randint
import pandas as pd
# seed random number generator
#seed(1)

TOTAL_PCK = 27
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')
test_datapath = '../dataset/processdata/dataset_Q23_mousedel_time_val'
test_set = FixDataset(0, test_datapath)
test_loader = DataLoader(dataset=test_set, batch_size=1, num_workers=0, collate_fn=collate_fn, shuffle=False)

time = 0
iter = 100
all_gaze = pd.DataFrame()
end_prob = 1 / 8.7 * 100
minLen = 1

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
    length = tgt_out.size()[0]

    for n in range(iter):
        GAZE = []
        x = randint(0, 101)
        while x >= end_prob or len(GAZE)<minLen:
            ind = randint(0, TOTAL_PCK-1)
            GAZE.append(ind)
            x = randint(0, 101)
        gaze_df = np.stack(GAZE).reshape(1, -1)
        all_gaze = pd.concat([all_gaze, pd.DataFrame(gaze_df)],axis=0)

loss = np.log(TOTAL_PCK+3)
print('loss=', loss)
all_gaze.to_csv('../dataset/checkEvaluation/gaze_random.csv', index=False)

