import numpy as np
from torch.utils.data import DataLoader
import sys
sys.path.append('dataBuilders')
from data_builder_mit1003 import MIT1003Dataset, Collator
import torch
from random import seed
from random import randint
sys.path.append('evaluation')
from evaluation_mit1003 import EvaluationMetric
from tqdm import tqdm

seed(1)

# TODO: carefully fit the number accordingly
iter = 1
TOTAL_PCK = 16
minLen = 10
metrics = EvaluationMetric()
DEVICE = torch.device('cpu')

SED = []
SBTDE = []

class ArgsRandom(object):
    def __init__(self,foldNum):
        self.fold =foldNum
        self.data_folder_path = '../dataset/MIT1003/'

for i in tqdm(range(1,11)):
    args = ArgsRandom(i)
    mit = MIT1003Dataset(args, False)
    collate_fn = Collator(mit.getImageData())
    test_loader = DataLoader(dataset=mit,
                                  batch_size=1,
                                  num_workers=0,
                                  collate_fn=collate_fn,
                                  shuffle=False)

    for src_pos, src_img, tgt_pos, tgt_img in test_loader:
        src_pos = src_pos.to(DEVICE)
        tgt_pos = tgt_pos.to(DEVICE)
        tgt_out = tgt_pos[1:, :]
        tgtLength = tgt_out.size()[0] - 1
        if tgtLength<minLen:
            continue
        else:
             for n in range(iter):
                Gaze = []
                while len(Gaze)<minLen:
                    ind = randint(0,TOTAL_PCK-1)
                    Gaze.append(ind)
                Gaze = np.stack(Gaze).reshape(-1)
                GT = tgt_out[:minLen,:].reshape(-1).numpy()
                sed = np.stack([metrics.string_edit_distance(GT[:i],Gaze[:i]) for i in range(1,minLen+1)]).mean()
                sbtde = np.stack([metrics.string_based_time_delay_embedding_distance(GT,Gaze,k) for k in range(1,minLen+1)]).mean()
                SED.append(sed)
                SBTDE.append(sbtde)

loss = np.log(TOTAL_PCK+3)
print('loss=', loss)
print('Mean SED:',np.stack(SED).mean())
print('Mean SBTDE:',np.stack(SBTDE).mean())

