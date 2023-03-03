import numpy as np
# sys.path.append('evaluation')
from src.evaluation.evaluation_mit1003 import EvaluationMetric
from tqdm import tqdm
import os
import pandas as pd
import cv2


def normalize_map(s_map):
	# normalize the salience map (as done in MIT code)
	norm_s_map = (s_map - np.min(s_map))/((np.max(s_map)-np.min(s_map))*1.0)
	return norm_s_map


def getFileList(dir, Filelist, ext=None):
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-3:]:
                Filelist.append(dir)

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getFileList(newDir, Filelist, ext)

    return Filelist

def random_uniform(n,x_max,y_max):
    xy_min = [1, 1]
    xy_max = [x_max, y_max]
    data = np.random.randint(low=xy_min, high=xy_max, size=(n, 2))
    return data


metrics = EvaluationMetric()
saliency_img_folder = '../dataset/MIT1003/SALIENCY_MAPS'
gazePath = '../dataset/MIT1003/MIT1003.xlsx'
gazesExcel = pd.read_excel(gazePath)
subList = ['ajs', 'CNG', 'emb','ems','ff','hp','jcw','jw','kae','krl','po','tmj','tu','ya','zb']
saliency_imglist = sorted(getFileList(saliency_img_folder, [], 'jpg'))
dataLength = len(saliency_imglist)
print('num of saliency_imglist ' + str(len(saliency_imglist)) + '\n')

auc_judd_score = []
nss_score = []
ll_score = []

for i in tqdm(range(dataLength)):
    saliency_imgPath = saliency_imglist[i]
    imgName = os.path.splitext(os.path.basename(saliency_imgPath))[0]
    s_map = cv2.imread(saliency_imgPath,0)
    s_map = normalize_map(s_map)

    for j in range(len(subList)):
        gt_sub = np.zeros((s_map.shape[0],s_map.shape[1]))
        base_sub = np.zeros((s_map.shape[0],s_map.shape[1]))
        current_sub = subList[j]

        df = gazesExcel[(gazesExcel.Sub == current_sub) & (gazesExcel.Task == imgName + '.jpeg')]
        if df.empty:
            pass
        else:
            X = list(df['X'])
            Y = list(df['Y'])
            eachLength = len(X)
            base_xy = random_uniform(eachLength,s_map.shape[0],s_map.shape[1])

            for m in range(eachLength):
                if 0<Y[m]<=s_map.shape[0] and 0<X[m]<=s_map.shape[1]:
                    gt_sub[Y[m]-1,X[m]-1] = 1
                    base_sub[base_xy[m][0] - 1, base_xy[m][1] - 1] = 1
            auc_judd_score.append(metrics.AUC_Judd(s_map, gt_sub))
            nss_score.append(metrics.NSS(s_map, gt_sub))
            ll_score.append(metrics.InfoGain(s_map,gt_sub,base_sub))
    print('auc judd:',np.array(auc_judd_score).mean())
    print('nss:',np.array(nss_score).mean())
    print('ll:', np.array(ll_score).mean())

















