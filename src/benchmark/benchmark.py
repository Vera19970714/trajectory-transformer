import numpy as np
from torch.utils.data import DataLoader
import torch
# generate random integer values
from random import seed
from random import randint
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax
import sys
sys.path.append('./src/')
from dataBuilders.data_builder import *
from evaluation.evaluation_full import Evaluation
from evaluation.saliency_metric import SIM
from tqdm import tqdm
import matplotlib.image
import pySaliencyMap
from scipy.ndimage.filters import gaussian_filter
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import random
   
class Benchmark(object):
    def __init__(self, training_dataset_choice, testing_dataset_choice, saveFolder, datapath, indexFile,minLen=1,
                 ITERATION=100):
        # datapath = './dataset/processdata/dataset_Q123_mousedel_time'
        # indexFile = './dataset/processdata/splitlist_all_time.txt'
        self.saveFolder = saveFolder
        self.training_dataset_choice = training_dataset_choice
        self.testing_dataset_choice = testing_dataset_choice
        self.ITERATION = ITERATION
        self.minLen = minLen

        raw_data = randsplit(datapath, indexFile, 'Test', testing_dataset_choice, training_dataset_choice)
        raw_data_train = randsplit(datapath, indexFile, 'Train', testing_dataset_choice, training_dataset_choice)

        self.data_length = len(raw_data)
        self.data_length_train = len(raw_data_train)
        print(F'len = {self.data_length}')
        self.id = []
        
        self.package_target = []
        self.question_img_feature = []
        self.package_seq = []
        for item in raw_data:
            sequence_list = [x - 1 for x in item['package_seq']]
            self.package_seq.append(sequence_list)
            self.id.append(item['id'])
            self.package_target.append(item['package_target'])
            self.question_img_feature.append(item['question_img_feature'])

        # for compute hyperparameter
        self.id_train = []
        self.package_seq_train = []
        for item in raw_data_train:
            sequence_list = [x - 1 for x in item['package_seq']]
            self.package_seq_train.append(sequence_list)
            self.id_train.append(item['id'])

    def plot_hist(self, train_list_wine, test_list_wine,train_list_yogurt, test_list_yogurt):
        plt.subplot(2, 1, 1)
        plt.hist(train_list_wine, bins='auto', alpha=0.2, label='train_wine')
        plt.hist(test_list_wine, bins='auto', alpha=0.2, label='test_wine')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.title('Histogram of Wine')
        plt.legend()

        # Saving the first plot
        plt.savefig('histogram_1.png')

        # Clearing the plot
        plt.clf()
        # Plotting the other two histograms in a separate plot
        plt.subplot(2, 1, 1)
        plt.hist(train_list_yogurt, bins='auto', alpha=0.2, label='train_yogurt')
        plt.hist(test_list_yogurt, bins='auto', alpha=0.2, label='test_yogurt')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.title('Histogram of Yogurt')
        plt.legend()

        # Saving the second plot
        plt.savefig('histogram_2.png')

    def debug_length(self):
        each_length_wine_train= []
        each_length_wine_test = []
        each_length_yogurt_train= []
        each_length_yogurt_test = []
        whole_length = []
        for i in tqdm(range(self.data_length_train)):
            if self.training_dataset_choice == 'pure':
                if self.testing_dataset_choice == 'wine':
                    each_length_wine_train.append(len(self.package_seq_train[i]))
                elif self.testing_dataset_choice == 'yogurt':
                    each_length_yogurt_train.append(len(self.package_seq_train[i]))
            elif self.training_dataset_choice == 'all':      
                if self.id_train[i] == 'Q1':
                    each_length_wine_train.append(len(self.package_seq_train[i]))
                elif self.id_train[i] == 'Q3':
                    each_length_yogurt_train.append(len(self.package_seq_train[i]))
            # whole_length.append(len(self.package_seq_train[i])
        for i in tqdm(range(self.data_length)):
            if self.training_dataset_choice == 'pure':
                if self.testing_dataset_choice == 'wine':
                    each_length_wine_test.append(len(self.package_seq[i]))
                elif self.testing_dataset_choice == 'yogurt':
                    each_length_yogurt_test.append(len(self.package_seq[i]))
            elif self.training_dataset_choice == 'all':      
                if self.id[i] == 'Q1':
                    each_length_wine_test.append(len(self.package_seq[i]))
                elif self.id[i] == 'Q3':
                    each_length_yogurt_test.append(len(self.package_seq[i]))
        whole_wine = each_length_wine_train+each_length_wine_test
        whole_yogurt = each_length_yogurt_train+each_length_yogurt_test
        random.shuffle(whole_wine)
        random.shuffle(whole_yogurt)
        wine_train = whole_wine[:int(len(whole_wine) * 8/9)]
        wine_test = whole_wine[int(len(whole_wine) * 8/9):]
        yogurt_train = whole_yogurt[:int(len(whole_yogurt) * 0.8)]
        yogurt_test = whole_yogurt[int(len(whole_yogurt) * 0.9):]
        print(len(wine_test))
        print(len(yogurt_test))
        print('wine_train:', np.mean(wine_train))
        print('wine_test:',np.mean(wine_test))
        print('yogurt_train:',np.mean(yogurt_train))
        print('yogurt_test:',np.mean(yogurt_test))
        self.plot_hist(wine_train, wine_test, yogurt_train, yogurt_test)

    def random(self, endPro, TOTAL_PCK):
        random_dist = np.ones(TOTAL_PCK) / TOTAL_PCK
        return self.sample_gaze_from_distri(endPro, random_dist, TOTAL_PCK)
        

    def hyper_cal(self, bandwidth, eps = 1e-20):
        TOTAL_PCK_wine = 22
        TOTAL_PCK_yogurt = 27
        rowNum_wine = 2
        columNum_wine = 11
        rowNum_yogurt = 3
        columNum_yogurt = 9
        each_length= []
        fixation_wine = np.zeros(TOTAL_PCK_wine)
        fixation_yogurt = np.zeros(TOTAL_PCK_yogurt)
        for i in tqdm(range(self.data_length_train)):
            if self.training_dataset_choice == self.testing_dataset_choice and self.testing_dataset_choice != 'all':
                if self.testing_dataset_choice == 'wine':
                    each_length.append(len(self.package_seq_train[i]))
                    fixation_wine[self.package_seq_train[i]] += 1
                elif self.testing_dataset_choice == 'yogurt':
                    each_length.append(len(self.package_seq_train[i]))
                    fixation_yogurt[self.package_seq_train[i]] += 1
            elif self.training_dataset_choice == self.testing_dataset_choice == 'all':
                each_length.append(len(self.package_seq_train[i]))
                if self.id_train[i] == 'Q1':
                    # print(self.package_seq[i])
                    fixation_wine[self.package_seq_train[i]] += 1
                elif self.id_train[i] == 'Q3':
                    fixation_yogurt[self.package_seq_train[i]] += 1
            else:
                print('not implemented')
                quit()
        avg_length =np.mean(each_length)
        fixation_wine = gaussian_filter(fixation_wine.reshape(rowNum_wine,columNum_wine), [bandwidth*rowNum_wine, bandwidth*columNum_wine])
        fixation_wine *= (1-eps)
        fixation_wine += eps * 1.0/(rowNum_wine*columNum_wine)
        fixation_wine = np.log(fixation_wine)
        fixation_wine -= logsumexp(fixation_wine)
        
        fixation_yogurt = gaussian_filter(fixation_yogurt.reshape(rowNum_yogurt,columNum_yogurt), [bandwidth*rowNum_yogurt, bandwidth*columNum_yogurt])
        fixation_yogurt *= (1-eps)
        fixation_yogurt += eps * 1.0/(rowNum_yogurt*columNum_yogurt)
        #plt.imshow(fixation_yogurt)
        fixation_yogurt = np.log(fixation_yogurt)
        fixation_yogurt -= logsumexp(fixation_yogurt)
        # todo: 1. this is not distribution, 2. gaussian filter not working, radius too small
        return avg_length, fixation_wine, fixation_yogurt
    

    def center(self, endPro, center_dis, TOTAL_PCK):
        # using https://github.com/Davidelanz/saliency_prediction/blob/main/CreateCenterbias.ipynb'''
        return self.sample_gaze_from_distri(endPro, center_dis, TOTAL_PCK)

    def sample_gaze_from_distri(self, avg_len, distri, TOTAL_PCK):
        end_prob = 1 / avg_len * 10000
        gaze = []
        x = randint(0, 10000)
        minLen = 1
        while x >= end_prob or len(gaze) < minLen:
            ind = np.random.choice(TOTAL_PCK, 1, p=distri)
            gaze.append(np.ndarray.item(ind))
            x = randint(0, 10000)
        gaze = np.stack(gaze).reshape(1, -1)
        return gaze


    def saliency(self, endPro, TOTAL_PCK,rowNum, columNum,img_feature):
        reshaped_array = np.array(img_feature).reshape(rowNum, columNum, img_feature[0].shape[0], img_feature[0].shape[1], img_feature[0].shape[2])
        large_image = np.concatenate(reshaped_array, axis=1)
        large_image = np.concatenate(large_image, axis=1)
        large_image[large_image < 0] = 0
        sm = pySaliencyMap.pySaliencyMap(large_image.shape[1], large_image.shape[0])
        saliency_map = sm.SMGetSM(large_image)
        feature_dis = []
        for y in range(rowNum):
            for x in range(columNum):
                img_cropped_saliency = saliency_map[(y*img_feature[0].shape[0]):(y+1)*img_feature[0].shape[0], (x*img_feature[0].shape[1]):(x+1)*img_feature[0].shape[1]]
                img_cropped_saliency_mean = np.mean(img_cropped_saliency)
                feature_dis.append(img_cropped_saliency_mean)
        # feature_dis = softmax(feature_dis).reshape(-1)
        feature_dis = feature_dis / np.sum(feature_dis)
        feature_dis = np.array(feature_dis).reshape(-1)
        gaze = self.sample_gaze_from_distri(endPro, feature_dis, TOTAL_PCK)
        return feature_dis, gaze

    def rgb_similarity(self, endPro, TOTAL_PCK, package_target, img_feature):
        target_img_feature = img_feature[package_target[0]-1]
        feature_dis = []
        length = len(img_feature)
        for i in range(length):
            current_img_feature = img_feature[i]
            feature_cos = cosine_similarity(target_img_feature.reshape(1,-1), current_img_feature.reshape(1,-1))
            feature_dis.append(feature_cos)
        # feature_dis = softmax(feature_dis).reshape(-1)
        feature_dis = feature_dis/ np.sum(feature_dis)
        feature_dis = np.array(feature_dis).reshape(-1)
        gaze = self.sample_gaze_from_distri(endPro, feature_dis, TOTAL_PCK)
        return feature_dis, gaze
    

    def benchmark(self):
        random_gaze, center_gaze, saliency_gaze, rgb_gaze= pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),pd.DataFrame()
        avg_length, fixation_wine, fixation_yogurt = self.hyper_cal(bandwidth = 0.0217) # todo: wrong bandwidth value
        #end_prob = 1 / (avg_length+1) * 100
        fixation_wine = fixation_wine.reshape(-1)
        fixation_yogurt = fixation_yogurt.reshape(-1)
        fixation_wine = fixation_wine / np.sum(fixation_wine) # todo: cannot divide sum of log density
        fixation_yogurt = fixation_yogurt / np.sum(fixation_yogurt)
        sim_random = []
        sim_center = []
        sim_saliency = []
        sim_rgb = []
        for i in tqdm(range(self.data_length)):
            dis_dict = {}
            if self.training_dataset_choice == self.testing_dataset_choice and self.testing_dataset_choice != 'all':
                if self.testing_dataset_choice == 'wine':
                    TOTAL_PCK = 22
                    rowNum = 2
                    columNum = 11
                    center_dis = fixation_wine
                elif self.testing_dataset_choice == 'yogurt':
                    TOTAL_PCK = 27
                    rowNum = 3
                    columNum = 9
                    center_dis = fixation_yogurt
            elif self.training_dataset_choice == self.testing_dataset_choice == 'all':
                if self.id[i] == 'Q1':
                    TOTAL_PCK = 22
                    rowNum = 2
                    columNum = 11
                    center_dis = fixation_wine
                elif self.id[i] == 'Q3':
                    TOTAL_PCK = 27
                    rowNum = 3
                    columNum = 9
                    center_dis = fixation_yogurt
                    
            package_target = self.package_target[i]
            img_feature = self.question_img_feature[i]
            gt = self.package_seq[i]
            heatmap_gt = np.zeros(TOTAL_PCK)
            heatmap_gt[np.array(gt)] = 1

            for n in range(self.ITERATION):
                random_gaze_each = self.random(avg_length,TOTAL_PCK)
                center_gaze_each = self.center(avg_length, center_dis, TOTAL_PCK)
                saliency_dis, saliency_gaze_each = self.saliency(avg_length,TOTAL_PCK,rowNum, columNum, img_feature)
                rgb_dis, rgb_gaze_each = self.rgb_similarity(avg_length,TOTAL_PCK, package_target,img_feature)
                
                random_gaze = pd.concat([random_gaze, pd.DataFrame(random_gaze_each)],axis=0)
                center_gaze = pd.concat([center_gaze, pd.DataFrame(center_gaze_each)],axis=0)
                saliency_gaze = pd.concat([saliency_gaze, pd.DataFrame(saliency_gaze_each)],axis=0)
                rgb_gaze = pd.concat([rgb_gaze, pd.DataFrame(rgb_gaze_each)],axis=0)
            # todo: compare the same distribution with heatmap generated from each (only) fixation
            sim_random.append(SIM(np.ones(TOTAL_PCK) / TOTAL_PCK,heatmap_gt))
            sim_center.append(SIM(center_dis,heatmap_gt))
            sim_saliency.append(SIM(saliency_dis,heatmap_gt))
            sim_rgb.append(SIM(rgb_dis,heatmap_gt))
        # print(random_gaze)
        # exit()
        random_gaze.to_csv(self.saveFolder + 'gaze_random.csv', index=False)
        center_gaze.to_csv(self.saveFolder + 'gaze_center.csv', index=False)
        saliency_gaze.to_csv(self.saveFolder + 'gaze_saliency.csv', index=False)
        rgb_gaze.to_csv(self.saveFolder + 'gaze_rgb_similarity.csv', index=False)

        print('sim_random:', np.mean(sim_random))
        print('sim_center:', np.mean(sim_center))
        print('sim_saliency:', np.mean(sim_saliency))
        print('sim_rgb:', np.mean(sim_rgb))
        

if __name__ == '__main__':
    '''training_dataset_choice = 'all'
    testing_dataset_choice = 'all'
    saveFolder = './dataset/checkEvaluation/'
    logFile = 'mixed_pe_exp1_alpha9'
    datapath = './dataset/processdata/dataset_Q123_mousedel_time'
    indexFile = './dataset/processdata/splitlist_all_time.txt'
    b = Benchmark(training_dataset_choice, testing_dataset_choice, saveFolder, datapath, indexFile)
    b.benchmark()
    e = Evaluation(training_dataset_choice, testing_dataset_choice, saveFolder+logFile, datapath, indexFile)
    e.evaluation()'''

    def sample_gaze_from_distri(avg_len, distri, TOTAL_PCK):
        end_prob = 1 / avg_len * 10000
        gaze = []
        x = randint(0, 10000)
        minLen = 1
        while x >= end_prob or len(gaze) < minLen:
            ind = np.random.choice(TOTAL_PCK, 1, p=distri)
            gaze.append(np.ndarray.item(ind))
            x = randint(0, 10000)
        gaze = np.stack(gaze).reshape(1, -1)
        return gaze

    TOTAL_PCK = 22
    avg_length = 7.23
    random_dist = np.ones(TOTAL_PCK) / TOTAL_PCK
    total_len = 0
    number = 30000
    for i in range(number):
        gaze = sample_gaze_from_distri(avg_length, random_dist, TOTAL_PCK)
        total_len += gaze.shape[1]
    print(total_len/number)
