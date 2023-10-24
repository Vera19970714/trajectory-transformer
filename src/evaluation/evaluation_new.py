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
from saliency_evaluation import *
import sys
sys.path.append('./src/')
from dataBuilders.data_builder import randsplit


def behavior(result_array, target, gaze):
    for i in range(len(gaze)):
        if len(gaze[i]) == 0:
            print('GAZE LENGTH IS ZERO')
            continue
        gaze_element = gaze[i][~np.isnan(gaze[i])]
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


def losses(heatmap_gt, gaze, result_array, TOTAL_PCK):
    for i in range(len(gaze)):
        single = gaze[i][~np.isnan(gaze[i])]
        heatmap_single = torch.zeros(TOTAL_PCK)
        for element in single:
            heatmap_single[int(element)] = 1
        heatmap_single = heatmap_single / heatmap_single.sum()
        # result_array[6] += np.minimum(heatmap_single, heatmap_gt).sum()
        result_array[6] += AUC_Judd(heatmap_single, heatmap_gt)
        result_array[7] += NSS(heatmap_single, heatmap_gt)
        
def benchmark_losses(saliency_dis,heatmap_gt,result_array,ITERATION):
    for i in range(ITERATION):
        result_array[6] += AUC_Judd(saliency_dis, heatmap_gt)
        result_array[7] += NSS(saliency_dis, heatmap_gt)


class Evaluation(object):
    def __init__(self, training_dataset_choice, testing_dataset_choice, evaluation_url,
                 ITERATION=100, showBenchmark=True):
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

        datapath = './dataset/processdata/dataset_Q123_mousedel_time'
        indexFile = './dataset/processdata/splitlist_all.txt'
        dispath = './dataset/processdata/benchmark_dis_time'
        
        raw_data = randsplit(datapath, indexFile, 'Test', testing_dataset_choice, training_dataset_choice)
        with open(dispath, "rb") as fp:
            self.dis_data = pickle.load(fp)

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
            center_dis = self.dis_data[i]['center_dis']
            saliency_dis = self.dis_data[i]['saliency_dis']
            rgb_dis = self.dis_data[i]['rgb_dis']

            behavior(res['gt'], self.target[i], self.gaze_gt[i:(i+1)])
            behavior(res['single'], self.target[i], self.gaze_max[i:(i + 1)])
            behavior(res['multi'], self.target[i], self.gaze_expect[(i * self.ITERATION):(i * self.ITERATION + self.ITERATION)])
            if self.showBenchmark:
                behavior(res['random'], self.target[i],self.gaze_random[(i * self.ITERATION):(i * self.ITERATION + self.ITERATION)])
                behavior(res['center'], self.target[i], self.gaze_center[(i * self.ITERATION):(i * self.ITERATION + self.ITERATION)])
                behavior(res['rgb'], self.target[i], self.gaze_rgb[(i * self.ITERATION):(i * self.ITERATION + self.ITERATION)])
                behavior(res['saliency'], self.target[i], self.gaze_saliency[(i * self.ITERATION):(i * self.ITERATION + self.ITERATION)])
            
            if self.training_dataset_choice == 'pure':
                if self.testing_dataset_choice == 'wine':
                    TOTAL_PCK = 22
                elif self.testing_dataset_choice == 'yogurt':
                    TOTAL_PCK = 27
            elif self.training_dataset_choice == 'mixed':
                if self.ids[i] == 'Q1':
                    TOTAL_PCK = 22
                elif self.ids[i] == 'Q3':
                    TOTAL_PCK = 27
           
            gt = self.gaze_gt[i][~np.isnan(self.gaze_gt[i])]
            heatmap_gt = torch.zeros(TOTAL_PCK)
            for element in gt:
                heatmap_gt[int(element)] = 1
            losses(heatmap_gt, self.gaze_max[i:(i + 1)], res['single'], TOTAL_PCK)
            losses(heatmap_gt, self.gaze_expect[(i * self.ITERATION):(i * self.ITERATION + self.ITERATION)], res['multi'], TOTAL_PCK)
            
            if self.showBenchmark:
                random_dis = np.ones(TOTAL_PCK) / TOTAL_PCK
                benchmark_losses(random_dis, heatmap_gt,res['random'],self.ITERATION)
                benchmark_losses(center_dis, heatmap_gt,res['center'],self.ITERATION)
                benchmark_losses(saliency_dis, heatmap_gt,res['saliency'],self.ITERATION)
                benchmark_losses(rgb_dis, heatmap_gt,res['rgb'],self.ITERATION)

        res['gt'] = res['gt'] / self.data_length
        res['single'] = res['single'] / self.data_length
        res['multi'] = res['multi'] / self.data_length / self.ITERATION
        if self.showBenchmark:
            res['random'] = res['random'] / self.data_length / self.ITERATION
            res['center'] = res['center'] / self.data_length / self.ITERATION
            res['rgb'] = res['rgb'] / self.data_length / self.ITERATION
            res['saliency'] = res['saliency'] / self.data_length / self.ITERATION

        print('*'*20)
        print('correct Target \t avg.len \t avg.search \t avg.refix \t avg.revisit \t delta \t AUC \t NSS')
        models = ['gt', 'single', 'multi']
        if self.showBenchmark:
            models.extend(['random','center', 'saliency', 'rgb'])
        for i in models:
            res[i][5] = torch.sum(torch.abs(res[i][:5] - res['gt'][:5]) / res['gt'][:5]) / 5
            print(i, ': ', res[i])
        print('*' * 20)


if __name__ == '__main__':
    ITERATION = 100
    training_dataset_choice = 'mixed'
    testing_dataset_choice = 'yogurt'

    evaluation_url = './dataset/checkEvaluation/mixed_pe_exp1_alpha9'

    e = Evaluation(training_dataset_choice, testing_dataset_choice, evaluation_url, ITERATION)
    e.evaluation()