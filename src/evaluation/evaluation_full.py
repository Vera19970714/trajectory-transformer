import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import torch.nn.functional as F
from itertools import groupby
from dictances import bhattacharyya
from scipy.stats import wasserstein_distance
import torch
import sys
sys.path.append('./src/')
from dataBuilders.data_builder import randsplit
from evaluation.saliency_metric import nw_matching
from evaluation.multimatch import docomparison


def behavior(result_array, target, gaze, benchmark=False):
    for i in range(len(gaze)):
        if len(gaze[i]) == 0:
            print('GAZE LENGTH IS ZERO')
            continue
        gaze_element = gaze[i][~np.isnan(gaze[i])]
        if benchmark:
            if len(gaze_element) > 1:
                gaze_element = gaze_element[:-1]
        if len(gaze_element) == 0:
            print('replacing it...')
            gaze_element = gaze[i-1][~np.isnan(gaze[i-1])]
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

def string_distance(result_array,gaze,gt,ITERATION,col_num,row_num):
    gt = gt[~np.isnan(gt)]
    for i in range(ITERATION):
        gaze_element = gaze[i][~np.isnan(gaze[i])]
        result_array[6] += nw_matching(gaze_element, gt)
        result_array[7] += np.mean(docomparison(gaze_element, gt,col_num, row_num))


class Evaluation(object):
    def __init__(self, training_dataset_choice, testing_dataset_choice, evaluation_url,
                 datapath, indexFile, ITERATION=100, showBenchmark=True):
        #gaze_tf = '../dataset/checkEvaluation/gaze_tf.csv'
        self.ITERATION = ITERATION
        self.showBenchmark = showBenchmark
        self.training_dataset_choice = training_dataset_choice
        self.testing_dataset_choice = testing_dataset_choice
        gaze_gt = evaluation_url+'/gaze_gt.csv'
        gaze_max = evaluation_url+'/gaze_max.csv'
        gaze_expect = evaluation_url+'/gaze_expect.csv'
        if showBenchmark:
            gaze_random = './dataset/checkEvaluation/gaze_random.csv'
            gaze_saliency = './dataset/checkEvaluation/gaze_saliency.csv'
            gaze_rgb = './dataset/checkEvaluation/gaze_rgb_similarity.csv'
            gaze_center = './dataset/checkEvaluation/gaze_center.csv'
        
        raw_data = randsplit(datapath, indexFile, 'Test', testing_dataset_choice, training_dataset_choice)
        

        self.data_length = len(raw_data)
        print(F'len = {self.data_length}')
        self.target = []
        self.ids = []

        for item in raw_data:
            self.target.append(item['package_target'])
            self.ids.append(item['id'])

        self.target = [int(self.target[i][0])-1 for i in range(len(self.target))]
  
        self.gaze_gt = np.array(pd.read_csv(gaze_gt))
        self.gaze_max = np.array(pd.read_csv(gaze_max))
        self.gaze_expect = np.array(pd.read_csv(gaze_expect))

        if showBenchmark:
            self.gaze_random = np.array(pd.read_csv(gaze_random))
            self.gaze_saliency = np.array(pd.read_csv(gaze_saliency))
            self.gaze_rgb = np.array(pd.read_csv(gaze_rgb))
            self.gaze_center = np.array(pd.read_csv(gaze_center))


    def evaluation(self):
        # 7 stands for: correct target, avg.length, avg.search, avg.refix, avg.revisit, distance, heatmap overlapping
        res = {'gt': torch.zeros(8), 'random': torch.zeros(8), 'center': torch.zeros(8),
               'rgb': torch.zeros(8), 'saliency': torch.zeros(8), #'tf': torch.zeros(5),
               'single': torch.zeros(8), 'multi': torch.zeros(8)}

        for i in range(self.data_length):
            if self.training_dataset_choice == 'pure':
                if self.testing_dataset_choice == 'wine':
                    TOTAL_PCK = 22
                    col_num = 11
                    row_num = 2
                elif self.testing_dataset_choice == 'yogurt':
                    TOTAL_PCK = 27
                    col_num = 9
                    row_num = 3
            elif self.training_dataset_choice == 'all':
                if self.ids[i] == 'Q1':
                    TOTAL_PCK = 22
                    col_num = 11
                    row_num = 2
                elif self.ids[i] == 'Q3':
                    TOTAL_PCK = 27
                    col_num = 9
                    row_num = 3
            behavior(res['gt'], self.target[i], self.gaze_gt[i:(i+1)])
            behavior(res['single'], self.target[i], self.gaze_max[i:(i + 1)])
            behavior(res['multi'], self.target[i], self.gaze_expect[(i * self.ITERATION):(i * self.ITERATION + self.ITERATION)])
            if self.showBenchmark:
                behavior(res['random'], self.target[i],self.gaze_random[(i * self.ITERATION):(i * self.ITERATION + self.ITERATION)],benchmark=True)
                behavior(res['center'], self.target[i], self.gaze_center[(i * self.ITERATION):(i * self.ITERATION + self.ITERATION)],benchmark=True)
                behavior(res['rgb'], self.target[i], self.gaze_rgb[(i * self.ITERATION):(i * self.ITERATION + self.ITERATION)],benchmark=True)
                behavior(res['saliency'], self.target[i], self.gaze_saliency[(i * self.ITERATION):(i * self.ITERATION + self.ITERATION)],benchmark=True)
                string_distance(res['random'],self.gaze_random[(i * self.ITERATION):(i * self.ITERATION + self.ITERATION)],self.gaze_gt[i:(i+1)],self.ITERATION,col_num,row_num)
                string_distance(res['center'],self.gaze_center[(i * self.ITERATION):(i * self.ITERATION + self.ITERATION)],self.gaze_gt[i:(i+1)],self.ITERATION,col_num,row_num)
                string_distance(res['rgb'],self.gaze_rgb[(i * self.ITERATION):(i * self.ITERATION + self.ITERATION)],self.gaze_gt[i:(i+1)],self.ITERATION,col_num,row_num)
                string_distance(res['saliency'],self.gaze_saliency[(i * self.ITERATION):(i * self.ITERATION + self.ITERATION)],self.gaze_gt[i:(i+1)],self.ITERATION,col_num,row_num)
                
        res['gt'] = res['gt'] / self.data_length
        res['single'] = res['single'] / self.data_length
        res['multi'] = res['multi'] / self.data_length / self.ITERATION
        if self.showBenchmark:
            res['random'] = res['random'] / self.data_length / self.ITERATION
            res['center'] = res['center'] / self.data_length / self.ITERATION
            res['rgb'] = res['rgb'] / self.data_length / self.ITERATION
            res['saliency'] = res['saliency'] / self.data_length / self.ITERATION

        print('*'*20)
        print('correct Target \t avg.len \t avg.search \t avg.refix \t avg.revisit \t delta \t SS \t MM')
        models = ['gt', 'single', 'multi']
        if self.showBenchmark:
            models.extend(['random','center', 'saliency', 'rgb'])
        for i in models:
            res[i][5] = torch.sum(torch.abs(res[i][:5] - res['gt'][:5]) / res['gt'][:5]) / 5
            print(i, ': ', res[i])
        print('*' * 20)


if __name__ == '__main__':
    ITERATION = 100
    training_dataset_choice = 'all'
    testing_dataset_choice = 'all'

    evaluation_url = './dataset/checkEvaluation/mixed_pe_exp1_alpha9'

    e = Evaluation(training_dataset_choice, testing_dataset_choice, evaluation_url, ITERATION)
    e.evaluation()