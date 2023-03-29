import numpy as np
import cv2
import pandas as pd
import math
import seaborn as sns
from tqdm import tqdm
from matplotlib import pyplot as plt

gazePath = './dataset/MIT1003/MIT1003.xlsx'
stimuliPath = './dataset/MIT1003/ALLSTIMULI/'
save_path = './dataset/MIT1003/gazePlot/'
patchNum = 16

gazesExcel = pd.read_excel(gazePath)
numOfRows = len(gazesExcel)
gazeHeatmap = np.zeros((patchNum,patchNum))
for i in tqdm(range(numOfRows)):
    row = gazesExcel.loc[i]
    subject = row['Sub']
    task = row['Task']
    index = row['T']
    x_coor = row['X']  # shape[1]
    y_coor = row['Y']  # shape[0]
    # save the current entry and start a new one
    if index == 1:
        imagePath = stimuliPath + task
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imageH = image.shape[0]
        imageW = image.shape[1]
    if x_coor > 0 and y_coor > 0 and x_coor < imageW and y_coor < imageH:
        x_coor1 = math.floor((x_coor / image.shape[1]) * patchNum)
        y_coor1 = math.floor((y_coor / image.shape[0]) * patchNum)
        gazeHeatmap[y_coor1,x_coor1] += 1
gazeHeatmap = 100*gazeHeatmap / gazeHeatmap.sum()
gazeHeatmap = gazeHeatmap
hp = sns.heatmap(data=gazeHeatmap,
                    vmin=0,
                    vmax=20,
                    cmap=plt.get_cmap('Greens'),
                    annot = True,
                    fmt = '.1f',
                    zorder = 2,
                    cbar = True
            )
# plt.show()
plt.savefig(save_path + 'gazePlot' + '_' + str(patchNum) + '.jpg')