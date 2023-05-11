import numpy as np
from torch.utils.data import DataLoader
import sys
sys.path.append('/Users/yujingling/Documents/GitHub/trajectory-transformer/src/')
from dataBuilders.data_builder_mit1003 import MIT1003Dataset, Collator
import torch
from random import seed
import cv2
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

seed(1)

minLen = 10
foldNum = 1
DEVICE = torch.device('cpu')
grid = 4
imagePath = '../dataset/MIT1003/ALLSTIMULI/'
predictGazePath = '../dataset/MIT1003/checkEvaluation/gaze_max.csv'
gtGazePath = '../dataset/MIT1003/checkEvaluation/gaze_gt.csv'
horizontalSaccadeX = []
horizontalSaccadeY = []
verticalSaccadeX = []
verticalSaccadeY = []

horizontalNum = 0
verticalNum = 0
countNum = 0
class ArgsSaccadeLength(object):
    def __init__(self,foldNum):
        self.fold =foldNum
        self.data_folder_path = '../dataset/MIT1003/'
        self.processed_data_name = 'processedData_3_sod'
        self.grid_partition = 4
        self.number_of_patches = 4
        self.add_salient_OD = 'True'
        self.batch_size = 1

args = ArgsSaccadeLength(foldNum)
mit = MIT1003Dataset(args, False)
collate_fn = Collator(mit.getImageData(), False, args.grid_partition, args.number_of_patches, args.add_salient_OD)
test_loader = DataLoader(dataset=mit,
                          batch_size=1,
                          num_workers=0,
                          collate_fn=collate_fn,
                          shuffle=False)

for imageName, src_pos, src_img, tgt_pos, tgt_img in test_loader:
    src_pos = src_pos.to(DEVICE)
    tgt_pos = tgt_pos.to(DEVICE)
    tgt_out = tgt_pos[1:, :]
    tgtLength = tgt_out.size()[0] - 1
    if tgtLength<minLen:
        continue
    else:
        gt = tgt_out[:minLen, :].reshape(-1).numpy()  # size: ndarray (10,)
        image = cv2.imread(imagePath+imageName)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imageH_original = image.shape[0]
        imageW_original = image.shape[1]
        gtX = gt % grid
        gtY = gt // grid
        delta_gtX = [abs(gtX[i+1]-gtX[i]) for i in range(len(gtX)-1)]
        delta_gtY = [abs(gtY[i + 1] - gtY[i]) for i in range(len(gtY)-1)]
        if imageH_original<imageW_original:
            horizontalSaccadeX.extend(delta_gtX)
            horizontalSaccadeY.extend(delta_gtY)
            horizontalNum += 1
        elif imageH_original>imageW_original:
            verticalSaccadeX.extend(delta_gtX)
            verticalSaccadeY.extend(delta_gtY)
            verticalNum += 1
    countNum += 1
print('horizontal Task Num:', horizontalNum)
print('vertical Task Num:', verticalNum)
fig = plt.figure()
fig1 = fig.add_subplot(121)
fig1.bar(x=['Horizontal Saccade','Vertical Saccade'], height=[np.array(horizontalSaccadeX).sum(), np.array(horizontalSaccadeY).sum()])
# fig1.gca().xaxis.set_major_locator(ticker.MultipleLocator(5))
fig1.set_title('Horizontal Image Saccade')

fig2 = fig.add_subplot(122)
fig2.bar(x=['Horizontal Saccade','Vertical Saccade'], height=[np.array(verticalSaccadeX).sum(), np.array(verticalSaccadeY).sum()])
# fig2.gca().xaxis.set_major_locator(ticker.MultipleLocator(5))
fig2.set_title('Vertical Image Saccade')

plt.show()


def drawGazeDistribution(gtGazePath, predictGazePath):
    gtGazesExcel = pd.read_csv(gtGazePath)
    predictGazeExcel = pd.read_csv(predictGazePath)
    numOfRows = len(gtGazesExcel)
    gtAllpositions = []
    predictAllpositions = []
    for i in tqdm(range(numOfRows)):
        gtrow = gtGazesExcel.iloc[i,:minLen]
        predictrow = predictGazeExcel.iloc[i,:minLen]
        gtAllpositions.extend(gtrow)
        predictAllpositions.extend(predictrow)
    plt.hist([gtAllpositions,predictAllpositions], bins=np.arange(0.5,20.5), label=['gt','predict'])
    plt.legend(loc='upper right')
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()

# drawGazeDistribution(gtGazePath, predictGazePath)





