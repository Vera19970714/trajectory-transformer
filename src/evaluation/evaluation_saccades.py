import pickle

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import torch.nn.functional as F
from itertools import groupby
from scipy.stats import wasserstein_distance
import torch
import sys
sys.path.append('./src/')
from dataBuilders.data_builder import randsplit, randsplit_comb
from tqdm import tqdm
#from tabulate import tabulate


def behavior(result_array, target, gaze, col_num,row_num):
    #behavior = {'correct':[],'length': [], 'search': [], 'refix': [], 'revisit': []}
    for i in range(len(gaze)):
        if len(gaze[i]) == 0:
            print('GAZE LENGTH IS ZERO')
            continue
        gaze_element = gaze[i][~np.isnan(gaze[i])]
        if len(gaze_element) == 0:
            print('replacing it...')
            gaze_element = gaze[i-1][~np.isnan(gaze[i-1])]
        for fix in range(gaze_element.shape[0]-1):
            a=gaze_element[fix]
            b=gaze_element[fix+1]

            x1, y1 = a // col_num, a % col_num
            x2, y2 = b // col_num, b % col_num
            dis = abs(x2-x1)+abs(y2-y1)
            result_array.append(dis)
            print()



class Evaluation(object):
    def __init__(self, training_dataset_choice, testing_dataset_choice, evaluation_url,
                 datapath, indexFile, ITERATION=100, showBenchmark=True, showExpected=True, leave_one_comb_out=0):
        #gaze_tf = '../dataset/checkEvaluation/gaze_tf.csv'
        self.ITERATION = ITERATION
        self.showBenchmark = showBenchmark
        self.training_dataset_choice = training_dataset_choice
        self.testing_dataset_choice = testing_dataset_choice
        gaze_gt = evaluation_url+'/gaze_gt.csv'
        gaze_max = evaluation_url+'/gaze_max.csv'
        gaze_expect = evaluation_url+'/gaze_expect.csv'
        
        #raw_data = randsplit(datapath, indexFile, 'Test', testing_dataset_choice, training_dataset_choice)

        if leave_one_comb_out == 0:
            raw_data = randsplit(datapath, indexFile, 'Test', testing_dataset_choice, training_dataset_choice)
        else:
            raw_data = randsplit_comb(datapath, indexFile, 'Test', testing_dataset_choice, training_dataset_choice,
                                      'Q3_10', 'T3_12')

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
        if showExpected:
            self.gaze_expect = np.array(pd.read_csv(gaze_expect))
        self.showExpected = showExpected

    def evaluation(self):
        res = []
        for i in tqdm(range(self.data_length)):
            if self.training_dataset_choice == self.testing_dataset_choice and self.testing_dataset_choice != 'all':
                if self.testing_dataset_choice == 'wine':
                    TOTAL_PCK = 22
                    col_num = 11
                    row_num = 2
                elif self.testing_dataset_choice == 'yogurt':
                    TOTAL_PCK = 27
                    col_num = 9
                    row_num = 3
                elif self.testing_dataset_choice == 'amazon':
                    TOTAL_PCK = 84
                    col_num = 14
                    row_num = 6
            elif self.training_dataset_choice == self.testing_dataset_choice == 'all':
                if self.ids[i] == 'Q1':
                    TOTAL_PCK = 22
                    col_num = 11
                    row_num = 2
                elif self.ids[i] == 'Q3':
                    TOTAL_PCK = 27
                    col_num = 9
                    row_num = 3
            else:
                print('not implemented')
                quit()
            #behavior(res['single'], self.target[i], self.gaze_max[i:(i + 1)],col_num,row_num)
            if self.showExpected:
                behavior(res, self.target[i],
                         self.gaze_expect[(i * self.ITERATION):(i * self.ITERATION + self.ITERATION)],col_num,row_num)
            else:
                behavior(res, self.target[i], self.gaze_gt[i:(i + 1)], col_num, row_num)
        plt.figure(figsize=(8, 7))
        matplotlib.rcParams.update({'font.size': 20})
        plt.hist(res, bins=[0, 1,2,3,4,5,6,7,8,9,10])
        plt.xlabel('Manhatten distance of the saccade')
        plt.ylabel('Count')
        plt.show()



if __name__ == '__main__':
    '''ITERATION = 100
    training_dataset_choice = 'amazon'
    testing_dataset_choice = 'amazon'
    datapath = './dataset/processdata/dataset_amazon'
    indexFile = './dataset/processdata/splitlist_all_amazon.txt'''
    training_dataset_choice = 'all'
    testing_dataset_choice = 'all'
    datapath = './dataset/processdata/dataset_Q123_mousedel_time'
    indexFile = './dataset/processdata/splitlist_all_time.txt'
    evaluation_url = './dataset/checkEvaluation/best_pamformer'

    e = Evaluation(training_dataset_choice, testing_dataset_choice, evaluation_url,
                 datapath, indexFile, ITERATION=100, showBenchmark=False, showExpected=True,
                   leave_one_comb_out=0)
    e.evaluation()
