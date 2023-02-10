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
from collections import defaultdict

seed(1)

# TODO: carefully fit the number accordingly
iter = 1
TOTAL_PCK = 16
minLen = 10
metrics = EvaluationMetric()
DEVICE = torch.device('cpu')

meanSED = []
meanSBTDE = []
sppSED = []
sppSBTDE = []

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
    sppSedList, sppSbtdeList = [], []
    sppSedDict, sppSbtdeDict = defaultdict(list),defaultdict(list)
    for imageName, src_pos, src_img, tgt_pos, tgt_img in test_loader:
        src_pos = src_pos.to(DEVICE)
        tgt_pos = tgt_pos.to(DEVICE)
        tgt_out = tgt_pos[1:, :]
        tgtLength = tgt_out.size()[0] - 1
        if tgtLength<minLen:
            continue
        else:

             for n in range(iter):
                sppSedIter,sppSbtdeIter = [],[]
                Gaze = []
                while len(Gaze)<minLen:
                    ind = randint(0,TOTAL_PCK-1)
                    Gaze.append(ind)
                Gaze = np.stack(Gaze).reshape(-1)
                GT = tgt_out[:minLen,:].reshape(-1).numpy()
                sed = np.stack([metrics.string_edit_distance(GT[:i],Gaze[:i]) for i in range(1,minLen+1)]).mean()
                sbtde = np.stack([metrics.string_based_time_delay_embedding_distance(GT,Gaze,k) for k in range(1,minLen+1)]).mean()
                meanSED.append(sed)
                meanSBTDE.append(sbtde)
                sppSedIter.append(sed)
                sppSbtdeIter.append(sbtde)
             sppSedList.append((imageName,np.stack(sppSedIter).mean()))
             sppSbtdeList.append((imageName,np.stack(sppSbtdeIter).mean()))
    for k, v in sppSedList:
        sppSedDict[k].append(v)
    for k, v in sppSbtdeList:
        sppSbtdeDict[k].append(v)
    sppSED.append([min(list(sppSedDict.values())[i]) for i in range(len(sppSedDict))])
    sppSBTDE.append([min(list(sppSbtdeDict.values())[i]) for i in range(len(sppSbtdeDict))])

loss = np.log(TOTAL_PCK+3)
print('loss=', loss)
print('Mean SED:',np.stack(meanSED).mean())
print('Mean SBTDE:',np.stack(meanSBTDE).mean())
print('Spp SED:',statistics.mean(sum(sppSED,[])).mean())
print('Spp SBTDE:',np.stack(sum(sppSBTDE,[])).mean())

