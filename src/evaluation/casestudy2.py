import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
sys.path.append('./src/')
from dataBuilders.data_builder import randsplit

def calculateLength(res, target_pos, target_id, gaze):
    # find the target, re calcualte the scanpath, record length under the target
    for i in range(len(gaze)):
        #gaze_i = gaze[i]
        if len(gaze[i]) == 0:
            print('GAZE LENGTH IS ZERO')
            continue
        gaze_i = gaze[i][~np.isnan(gaze[i])]
        if target_pos != gaze_i[-1]:
            continue
        ind = np.where(gaze_i == target_pos)[0][0]
        '''if len(ind) == 0:
            continue
        else:
            ind = ind[0]'''
        gaze_i = gaze_i[:(ind+1)]
        if target_id not in res:
            res[target_id] = []
        res[target_id].append(len(gaze_i))

class Evaluation(object):
    def __init__(self, training_dataset_choice, testing_dataset_choice, evaluation_url,
                 ITERATION=100, TOTAL_PCK=27):
        self.ITERATION = ITERATION
        self.TOTAL_PCK = TOTAL_PCK
        index_folder = './dataset/processdata/'
        gaze_gt = evaluation_url+'/gaze_gt.csv'
        gaze_max = evaluation_url+'/gaze_max.csv'
        gaze_expect = evaluation_url+'/gaze_expect.csv'

        datapath = './dataset/processdata/dataset_Q123_mousedel_time'
        indexFile = './dataset/processdata/splitlist_all_time.txt'

        raw_data = randsplit(datapath, indexFile, 'Test', testing_dataset_choice, training_dataset_choice)

        self.data_length = len(raw_data)
        print(F'len = {self.data_length}')
        self.target = []
        self.target_id = []

        for item in raw_data:
            self.target.append(item['package_target'])
            self.target_id.append(item['tgt_id'])

        self.target = [int(self.target[i][0])-1 for i in range(len(self.target))]

        self.gaze_gt = np.array(pd.read_csv(gaze_gt))
        self.gaze_max = np.array(pd.read_csv(gaze_max))
        self.gaze_expect = np.array(pd.read_csv(gaze_expect))

    def evaluation(self, choice): # gt, max, expect
        self.res = {}

        for i in range(self.data_length):
            if choice == 0:
                calculateLength(self.res, self.target[i], self.target_id[i], self.gaze_gt[i:(i + 1)])
            elif choice == 1:
                calculateLength(self.res, self.target[i], self.target_id[i], self.gaze_max[i:(i + 1)])
            elif choice == 2:
                calculateLength(self.res, self.target[i], self.target_id[i], self.gaze_expect[(i * self.ITERATION):(i * self.ITERATION + self.ITERATION)])

        length_dic = {}
        for (key, value) in enumerate(self.res):
            res_list = self.res[value]
            length_dic[value] = np.mean(res_list)
        print()

if __name__ == '__main__':
    ITERATION = 100
    TOTAL_PCK = 22
    training_dataset_choice = 'pure'
    testing_dataset_choice = 'yogurt'

    evaluation_url = './dataset/checkEvaluation/yogurt_pure_full'

    e = Evaluation(training_dataset_choice, testing_dataset_choice, evaluation_url, ITERATION, TOTAL_PCK)
    e.evaluation(2)
