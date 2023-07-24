import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import torch.nn.functional as F
from itertools import groupby
from dictances import bhattacharyya
from scipy.stats import wasserstein_distance


ITERATION = 100
TOTAL_PCK = 27
cross_dataset = 'Cross' # v2 choices: None, Pure, Mixed, Cross
testing_dataset_choice = 'shampoo'# choices: yogurt, shampoo
index_folder = './dataset/processdata/'

def randsplit(file, indexFile, isTrain, cross_dataset):
    with open(file, "rb") as fp:
        raw_data = pickle.load(fp)
    data_length = len(raw_data)

    with open(indexFile) as f:
        lines = f.readlines()
    linesInt = [int(x) for x in lines]

    if cross_dataset == 'None':
        split_num = int(data_length*0.9)
    elif cross_dataset == 'No':
        split_num = 453

    if isTrain:
        train_index = np.array(linesInt[:split_num])
        traindata = np.array(raw_data)[train_index.astype(int)]
        return traindata
    else:
        test_index = np.array(linesInt[-(data_length - split_num):])
        valdata = np.array(raw_data)[test_index.astype(int)]
        return valdata


def cross_data_split(file, isTrain):
    with open(file, "rb") as fp:
        raw_data = pickle.load(fp)
    shampoo_task = []
    yogurt_task = []
    for index in range(len(raw_data)):
        if raw_data[index]['id'] == 'Q2':
            shampoo_task.append(index)
        elif raw_data[index]['id'] == 'Q3':
            yogurt_task.append(index)
    if isTrain:
        train_index = np.array(shampoo_task)
        traindata = np.array(raw_data)[train_index.astype(int)]
        return traindata
    else:
        val_index = np.array(yogurt_task)
        valdata = np.array(raw_data)[val_index.astype(int)]
        return valdata

def cross_data_split2(file, isTrain, indexFolder, crossChoice, testing_dataset_choice):
    with open(file, "rb") as fp:
        raw_data = pickle.load(fp)
    shampoo_task = []
    yogurt_task = []
    for index in range(len(raw_data)):
        if raw_data[index]['id'] == 'Q2':
            shampoo_task.append(index)
        elif raw_data[index]['id'] == 'Q3':
            yogurt_task.append(index)

    if isTrain:
        if crossChoice == 'Pure':
            with open(indexFolder + 'splitlist_' + testing_dataset_choice + '_pure_indices.txt') as f:
                lines = f.readlines()
            train_index = np.array([int(x[:-1]) for x in lines])
        elif crossChoice == 'Mixed':
            with open(indexFolder + 'splitlist_' + testing_dataset_choice + '_mixed_indices.txt') as f:
                lines = f.readlines()
            train_index = np.array([int(x[:-1]) for x in lines])
        elif crossChoice == 'Cross':
            with open(indexFolder + 'splitlist_' + testing_dataset_choice + '_cross_indices.txt') as f:
                lines = f.readlines()
            train_index = np.array([int(x[:-1]) for x in lines])
        traindata = np.array(raw_data)[train_index]
        return traindata
    else:
        with open(indexFolder + 'splitlist_' + testing_dataset_choice + '_testing_indices.txt') as f:
            lines = f.readlines()
        val_index = np.array([int(x[:-1]) for x in lines])
        valdata = np.array(raw_data)[val_index]
        return valdata


def behavior(result_array, target, gaze):
    for i in range(len(gaze)):
        gaze_element = gaze[i][~np.isnan(gaze[i])]
        result_array[0] += int(target == gaze_element[-1])
        result_array[1] += len(gaze_element)
        search_len = 0
        refix_len = 0
        revisit_len = 0
        previous_visited = []
        for k, g in groupby(gaze_element):
            subiterator_len = len(list(g))
            if k in previous_visited:
                revisit_len += 1
            else:
                search_len += 1
            if subiterator_len > 1:
                refix_len += (subiterator_len - 1)
            previous_visited.append(k)
        assert search_len + refix_len + revisit_len == len(gaze_element)
        result_array[2] += (search_len / len(gaze_element))
        result_array[3] += (refix_len / len(gaze_element))
        result_array[4] += (revisit_len / len(gaze_element))


def losses(heatmap_gt, gaze, result_array):
    for i in range(len(gaze)):
        single = gaze[i][~np.isnan(gaze[i])]
        heatmap_single = torch.zeros(TOTAL_PCK)
        for element in single:
            heatmap_single[int(element)] += 1
        heatmap_single = heatmap_single / heatmap_single.sum()
        distance = wasserstein_distance(heatmap_single, heatmap_gt)
        result_array[5] += distance
        result_array[6] += np.minimum(heatmap_single, heatmap_gt).mean()


class Evaluation(object):
    def __init__(self):
        #gaze_tf = '../dataset/checkEvaluation/gaze_tf.csv'
        gaze_gt = './dataset/checkEvaluation/cross_shampoo_cross/gaze_gt.csv'
        gaze_max = './dataset/checkEvaluation/cross_shampoo_cross/gaze_max.csv'
        gaze_expect = './dataset/checkEvaluation/cross_shampoo_cross/gaze_expect.csv'
        gaze_random = './dataset/checkEvaluation/gaze_random.csv'
        gaze_resnet = './dataset/checkEvaluation/gaze_resnet_similarity.csv'
        gaze_saliency = './dataset/checkEvaluation/gaze_saliency.csv'
        gaze_rgb = './dataset/checkEvaluation/gaze_rgb_similarity.csv'

        new_datapath = './dataset/processdata/dataset_Q23_mousedel_time'
        indexFile = './dataset/processdata/splitlist_time_mousedel.txt'
        if cross_dataset == 'None':
            raw_data = randsplit(new_datapath, indexFile, False, cross_dataset)
        else:
            raw_data = cross_data_split2(new_datapath, False, index_folder, cross_dataset, testing_dataset_choice)

        self.data_length = len(raw_data)
        print(F'len = {self.data_length}')
        self.target = []

        for item in raw_data:
            self.target.append(item['package_target'])
        
        self.target = [int(self.target[i][0])-1 for i in range(len(self.target))]
        
        self.gaze_gt = np.array(pd.read_csv(gaze_gt))
        #self.gaze_tf = np.array(pd.read_csv(gaze_tf))
        self.gaze_max = np.array(pd.read_csv(gaze_max))
        self.gaze_expect = np.array(pd.read_csv(gaze_expect))
        self.gaze_random = np.array(pd.read_csv(gaze_random))
        self.gaze_resnet = np.array(pd.read_csv(gaze_resnet))
        self.gaze_saliency = np.array(pd.read_csv(gaze_saliency))
        self.gaze_rgb = np.array(pd.read_csv(gaze_rgb))

    def evaluation(self):
        # 7 stands for: correct target, avg.length, avg.search, avg.refix, avg.revisit, distance, heatmap overlapping
        res = {'gt': torch.zeros(7), 'random': torch.zeros(7), 'resnet': torch.zeros(7),
               'rgb': torch.zeros(7), 'saliency': torch.zeros(7), #'tf': torch.zeros(5),
               'single': torch.zeros(7), 'multi': torch.zeros(7)}

        for i in range(self.data_length):
            behavior(res['gt'], self.target[i], self.gaze_gt[i:(i+1)])
            behavior(res['single'], self.target[i], self.gaze_max[i:(i + 1)])
            behavior(res['random'], self.target[i], self.gaze_random[(i * ITERATION):(i * ITERATION + ITERATION)])
            behavior(res['resnet'], self.target[i], self.gaze_resnet[(i * ITERATION):(i * ITERATION + ITERATION)])
            behavior(res['rgb'], self.target[i], self.gaze_rgb[(i * ITERATION):(i * ITERATION + ITERATION)])
            behavior(res['saliency'], self.target[i], self.gaze_saliency[(i * ITERATION):(i * ITERATION + ITERATION)])
            behavior(res['multi'], self.target[i], self.gaze_expect[(i * ITERATION):(i * ITERATION + ITERATION)])

            gt = self.gaze_gt[i][~np.isnan(self.gaze_gt[i])]
            heatmap_gt = torch.zeros(TOTAL_PCK)
            for element in gt:
                heatmap_gt[int(element)] += 1
            heatmap_gt = heatmap_gt / heatmap_gt.sum()

            losses(heatmap_gt, self.gaze_max[i:(i + 1)], res['single'])
            losses(heatmap_gt, self.gaze_random[(i * ITERATION):(i * ITERATION + ITERATION)], res['random'])
            losses(heatmap_gt, self.gaze_resnet[(i * ITERATION):(i * ITERATION + ITERATION)], res['resnet'])
            losses(heatmap_gt, self.gaze_rgb[(i * ITERATION):(i * ITERATION + ITERATION)], res['rgb'])
            losses(heatmap_gt, self.gaze_saliency[(i * ITERATION):(i * ITERATION + ITERATION)], res['saliency'])
            losses(heatmap_gt, self.gaze_expect[(i * ITERATION):(i * ITERATION + ITERATION)], res['multi'])

        res['gt'] = res['gt'] / self.data_length
        res['single'] = res['single'] / self.data_length
        res['random'] = res['random'] / self.data_length / ITERATION
        res['resnet'] = res['resnet'] / self.data_length / ITERATION
        res['rgb'] = res['rgb'] / self.data_length / ITERATION
        res['saliency'] = res['saliency'] / self.data_length / ITERATION
        res['multi'] = res['multi'] / self.data_length / ITERATION

        print('*'*20)
        print('correct Target \t avg.len \t avg.search \t avg.refix \t avg.revisit \t distance \t overlap')
        for i in ['gt', 'random', 'resnet', 'rgb', 'saliency', 'single', 'multi']:
            print(i, ': ', res[i])
        print('*' * 20)


if __name__ == '__main__':
    e = Evaluation()
    e.evaluation()