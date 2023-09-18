import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import torch.nn.functional as F
from itertools import groupby
from dictances import bhattacharyya
from scipy.stats import wasserstein_distance
import sys
sys.path.append('./src/') 
from dataBuilders.data_builder import randsplit, cross_data_split2, cross_data_split3

ITERATION = 100
TOTAL_PCK = 27
cross_dataset = 'Combine' # v2 choices: None, Pure, Mixed, Cross, Combine,Only
testing_dataset_choice = 'yogurt'# choices: yogurt, shampoo
index_folder = './dataset/processdata/'

def behavior(result_array, target, gaze):
    print(len(gaze))
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
        result_array[6] += np.minimum(heatmap_single, heatmap_gt).sum()


class Evaluation(object):
    def __init__(self):
        #gaze_tf = '../dataset/checkEvaluation/gaze_tf.csv'
        gaze_gt = './dataset/checkEvaluation/combine_yogurt/gaze_gt.csv'
        gaze_max = './dataset/checkEvaluation/combine_yogurt/gaze_max.csv'
        gaze_expect = './dataset/checkEvaluation/combine_yogurt/gaze_expect.csv'
        gaze_random = './dataset/checkEvaluation/gaze_random.csv'
        gaze_resnet = './dataset/checkEvaluation/gaze_resnet_similarity.csv'
        gaze_saliency = './dataset/checkEvaluation/gaze_saliency.csv'
        gaze_rgb = './dataset/checkEvaluation/gaze_rgb_similarity.csv'

        new_datapath = './dataset/processdata/dataset_Q23_mousedel_time'
        indexFile = './dataset/processdata/splitlist_time_mousedel.txt'
        if cross_dataset == 'None':
            raw_data = randsplit(new_datapath, indexFile, 'Test', cross_dataset)
        elif cross_dataset == 'Only':
            raw_data = cross_data_split3(new_datapath, 'Test', index_folder, cross_dataset, testing_dataset_choice)
        else:
            raw_data = cross_data_split2(new_datapath, 'Test', index_folder, cross_dataset, testing_dataset_choice)

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
        res = {'gt': torch.zeros(8), 'random': torch.zeros(8), 'resnet': torch.zeros(8),
               'rgb': torch.zeros(8), 'saliency': torch.zeros(8), #'tf': torch.zeros(5),
               'single': torch.zeros(8), 'multi': torch.zeros(8)}

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
        print('correct Target \t avg.len \t avg.search \t avg.refix \t avg.revisit \t distance \t overlap \t delta')
        for i in ['gt', 'random', 'resnet', 'rgb', 'saliency', 'single', 'multi']:
            res[i][7] = torch.sum(torch.abs(res[i][:5] - res['gt'][:5]) / res['gt'][:5]) * 100
            print(i, ': ', res[i])
        print('*' * 20)


if __name__ == '__main__':
    e = Evaluation()
    e.evaluation()