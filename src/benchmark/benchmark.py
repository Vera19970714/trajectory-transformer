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
from skimage.transform import resize
from random import sample 
import copy
   
class Benchmark(object):
    def __init__(self, training_dataset_choice, testing_dataset_choice, saveFolder, processFolder, datapath, indexFile,minLen=1,
                 ITERATION=100):
        # datapath = './dataset/processdata/dataset_Q123_mousedel_time'
        # indexFile = './dataset/processdata/splitlist_all_time.txt'
        self.saveFolder = saveFolder
        self.filename = processFolder
        self.training_dataset_choice = training_dataset_choice
        self.testing_dataset_choice = testing_dataset_choice
        self.ITERATION = ITERATION
        self.minLen = minLen
        self.img_size =[1680,1050]
        self.crop_area_wine = [1680,152]
        self.crop_area_yogurt = [1680,135]


        raw_data = randsplit(datapath, indexFile, 'Test', testing_dataset_choice, training_dataset_choice)
        raw_data_train = randsplit(datapath, indexFile, 'Train', testing_dataset_choice, training_dataset_choice)

        self.data_length = len(raw_data)
        self.data_length_train = len(raw_data_train)
        print(F'len = {self.data_length}')
        self.id = []
        self.pair_test = []

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
        self.X_train = []
        self.Y_train = []
        for item in raw_data_train:
            sequence_list = [x - 1 for x in item['package_seq']]
            self.package_seq_train.append(sequence_list)
            self.id_train.append(item['id'])
            self.X_train.append(item['X'])
            self.Y_train.append(item['Y'])

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
        whole_pair = []
        whole_id = []
        for i in tqdm(range(self.data_length_train)):
            whole_length.append(len(self.package_seq_train[i]))  
            if self.training_dataset_choice == 'pure':
                if self.testing_dataset_choice == 'wine':
                    each_length_wine_train.append(len(self.package_seq_train[i]))
                elif self.testing_dataset_choice == 'yogurt':
                    each_length_yogurt_train.append(len(self.package_seq_train[i]))
            elif self.training_dataset_choice == 'all': 
                whole_id.append(self.id_train[i])
                whole_pair.append(self.pair_train[i])
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
                whole_id.append(self.id[i])
                whole_pair.append(self.pair_test[i])
                whole_length.append(len(self.package_seq[i]))     
                if self.id[i] == 'Q1':
                    each_length_wine_test.append(len(self.package_seq[i]))
                elif self.id[i] == 'Q3':
                    each_length_yogurt_test.append(len(self.package_seq[i]))
        print(np.mean(whole_length))
        print(np.mean(np.unique(whole_pair,return_counts=True)[1]))
        
        

    def hyper_cal(self, filename,bandwidth, eps = 1e-20):
        each_length= []
        fixation_wine = np.zeros((self.img_size[1],self.img_size[0]))
        fixation_yogurt = np.zeros((self.img_size[1],self.img_size[0]))

        for i in tqdm(range(self.data_length_train)):
            if self.training_dataset_choice == self.testing_dataset_choice and self.testing_dataset_choice != 'all':
                if self.testing_dataset_choice == 'wine':
                    each_length.append(len(self.package_seq_train[i]))
                    for x, y in zip(self.X_train[i], self.Y_train[i]):
                        fixation_wine[int(y),int(x)] += 1
                elif self.testing_dataset_choice == 'yogurt':
                    each_length.append(len(self.package_seq_train[i]))
                    for x, y in zip(self.X_train[i], self.Y_train[i]):
                        fixation_yogurt[int(y),int(x)] += 1
            elif self.training_dataset_choice == self.testing_dataset_choice == 'all':
                each_length.append(len(self.package_seq_train[i]))
                if self.id_train[i] == 'Q1':
                    for x, y in zip(self.X_train[i], self.Y_train[i]):
                        fixation_wine[int(y),int(x)] += 1
                elif self.id_train[i] == 'Q3':
                    for x, y in zip(self.X_train[i], self.Y_train[i]):
                        fixation_yogurt[int(y),int(x)] += 1
            else:
                print('not implemented')
                quit()
        avg_length =np.mean(each_length)
        '''fixation_wine = gaussian_filter(fixation_wine, [bandwidth*self.img_size[1], bandwidth*self.img_size[0]])
        fixation_wine *= (1-eps)
        fixation_wine += eps * 1.0/(self.img_size[0]*self.img_size[1])

        fixation_yogurt = gaussian_filter(fixation_yogurt, [bandwidth*self.img_size[1], bandwidth*self.img_size[0]])
        fixation_yogurt *= (1-eps)
        fixation_yogurt += eps * 1.0/(self.img_size[0]*self.img_size[1])
        np.save(filename + 'centerbias_wine.npy', fixation_wine)
        np.save(filename + 'centerbias_yogurt.npy', fixation_yogurt)'''
        return avg_length

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
    
    def pixel2region(self,centerbias,width,height,rowNum,columNum):
        new_array = np.zeros((rowNum,columNum))
        for i in range(rowNum):
            for j in range(columNum):
                region = centerbias[i*height:(i+1)*height,j*width:(j+1)*width]
                new_array[i,j] = np.sum(region)
        return new_array.reshape(-1)

    def sample_gaze_from_pixelDistri(self, avg_len, distri, width,height,columNum):
        end_prob = 1 / avg_len * 10000
        gaze = []
        x = randint(0, 10000)
        minLen = 1
        prob_dist = distri.flatten() 
        while x >= end_prob or len(gaze) < minLen:
            sampled_indices = np.random.choice(np.arange(len(prob_dist)), size=1, replace=False, p=prob_dist)
            sampled_coordinates = np.unravel_index(sampled_indices, distri.shape)
            ind = (sampled_coordinates[1] // width)+ (sampled_coordinates[0] // height)*columNum 
            gaze.append(np.ndarray.item(ind))
            x = randint(0, 10000)
        gaze = np.stack(gaze).reshape(1, -1)
        return gaze
    
    def sample_gaze_from_pixelDistri(self, avg_len, distri, width,height,columNum):
        end_prob = 1 / avg_len * 10000
        gaze = []
        x = randint(0, 10000)
        minLen = 1
        prob_dist = distri.flatten() 
        while x >= end_prob or len(gaze) < minLen:
            sampled_indices = np.random.choice(np.arange(len(prob_dist)), size=1, replace=False, p=prob_dist)
            sampled_coordinates = np.unravel_index(sampled_indices, distri.shape)
            ind = (sampled_coordinates[1] // width)+ (sampled_coordinates[0] // height)*columNum 
            gaze.append(np.ndarray.item(ind))
            x = randint(0, 10000)
        gaze = np.stack(gaze).reshape(1, -1)
        return gaze

    
    def saliency_dis(self,rowNum, columNum,img_feature):
        reshaped_array = np.array(img_feature).reshape(rowNum, columNum, img_feature[0].shape[0], img_feature[0].shape[1], img_feature[0].shape[2])
        large_image = np.concatenate(reshaped_array, axis=1)
        large_image = np.concatenate(large_image, axis=1)
        large_image[large_image < 0] = 0
        sm = pySaliencyMap.pySaliencyMap(large_image.shape[1], large_image.shape[0])
        saliency_map = sm.SMGetSM(large_image)
        '''feature_dis = []
        for y in range(rowNum):
            for x in range(columNum):
                img_cropped_saliency = saliency_map[(y*img_feature[0].shape[0]):(y+1)*img_feature[0].shape[0], (x*img_feature[0].shape[1]):(x+1)*img_feature[0].shape[1]]
                img_cropped_saliency_mean = np.mean(img_cropped_saliency)
                feature_dis.append(img_cropped_saliency_mean)
        # feature_dis = softmax(feature_dis).reshape(-1)
        feature_dis = feature_dis / np.sum(feature_dis)
        feature_dis = np.array(feature_dis).reshape(-1)'''
        return saliency_map

    def rgb_similarity_dis(self,package_target, img_feature):
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
        return feature_dis
    
    def winner_takes_all(self,input_saliency_map, avg_len, width,height,columNum,remove_radius = 40):
        end_prob = 1 / avg_len * 10000
        gaze = []
        x = randint(0, 10000)
        minLen = 1
        dis_list = []
        while x >= end_prob or len(gaze) < minLen:
            dis_list.append(input_saliency_map)
            y_max, x_max = np.shape(input_saliency_map)
            max_value = input_saliency_map.max()
            max_indexs = list(zip(*np.where(input_saliency_map == max_value))) # list
            max_index = sample(max_indexs,1)
            y,x = max_index[0]
            x_1 = x - remove_radius
            if x_1 < 0:
                x_1 = 0
            y_1 = y - remove_radius
            if y_1 < 0:
                y_1 = 0
            x_2 = x + remove_radius + 1
            if x_2 > x_max:
                x_2 = x_max
            y_2 = y + remove_radius + 1 
            if y_2 > y_max:
                y_2 = y_max
            input_saliency_map[y_1:y_2 , x_1:x_2] = 0
            ind = (max_index[0][1] // width)+ (max_index[0][0] // height)*columNum 
            gaze.append(ind)
            x = randint(0, 10000)
        gaze = np.stack(gaze).reshape(1, -1)
        return gaze


    def compute_sim(self,saliency_map,gaze,package_size):
        sim_total = 0
        seq_len = len(gaze)
        for i in range(seq_len):
            fixation_map = np.zeros(package_size)
            fixation_map[gaze[i]] = 1
            sim = SIM(saliency_map, fixation_map)
            sim_total += sim
        return sim_total/seq_len
   
    def benchmark(self):
        random_gaze, center_gaze, saliency_gaze, rgb_gaze= pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),pd.DataFrame()
        avg_length = self.hyper_cal(self.filename,bandwidth = 0.0434) 
        sim_random = []
        sim_center = []
        sim_saliency = []
        sim_rgb = []
        centerbias_wine = np.load(self.filename + 'centerbias_wine.npy')
        centerbias_yogurt = np.load(self.filename + 'centerbias_yogurt.npy')
        centerbias_wine = centerbias_wine[self.crop_area_wine[1]:,:]
        centerbias_yogurt = centerbias_yogurt[self.crop_area_wine[1]:,:]
        centerbias_wine = centerbias_wine / np.sum(centerbias_wine) 
        centerbias_yogurt = centerbias_yogurt / np.sum(centerbias_yogurt)
        for i in tqdm(range(self.data_length)):
            if self.training_dataset_choice == self.testing_dataset_choice and self.testing_dataset_choice != 'all':
                if self.testing_dataset_choice == 'wine':
                    TOTAL_PCK = 22
                    rowNum = 2
                    columNum = 11
                    IMAGE_SIZE_1 = 449
                    IMAGE_SIZE_2 = 152
                    centerbias = centerbias_wine
                elif self.testing_dataset_choice == 'yogurt':
                    TOTAL_PCK = 27
                    rowNum = 3
                    columNum = 9
                    IMAGE_SIZE_1 = 305
                    IMAGE_SIZE_2 = 186
                    centerbias = centerbias_yogurt
            elif self.training_dataset_choice == self.testing_dataset_choice == 'all':
                if self.id[i] == 'Q1':
                    TOTAL_PCK = 22
                    rowNum = 2
                    columNum = 11
                    IMAGE_SIZE_1 = 449
                    IMAGE_SIZE_2 = 152
                    centerbias = centerbias_wine
                elif self.id[i] == 'Q3':
                    TOTAL_PCK = 27
                    rowNum = 3
                    columNum = 9
                    IMAGE_SIZE_1 = 305
                    IMAGE_SIZE_2 = 186
                    centerbias = centerbias_yogurt
            package_target = self.package_target[i]
            img_feature = self.question_img_feature[i]
            gt = self.package_seq[i]
            center_dis = self.pixel2region(centerbias, IMAGE_SIZE_2, IMAGE_SIZE_1,rowNum,columNum)
            '''saliency_dis = self.saliency_dis(rowNum, columNum,img_feature)
            rgb_dis = self.rgb_similarity_dis(package_target, img_feature)
            input_saliency_map = copy.deepcopy(saliency_dis)'''
            for n in range(self.ITERATION):
                # random_gaze_each = self.sample_gaze_from_distri(avg_length,np.ones(TOTAL_PCK) / TOTAL_PCK, TOTAL_PCK)
                center_gaze_each = self.sample_gaze_from_pixelDistri(avg_length, centerbias, IMAGE_SIZE_2,IMAGE_SIZE_1,columNum)
                # saliency_gaze_each = self.sample_gaze_from_distri(avg_length, saliency_dis, TOTAL_PCK)
                # saliency_gaze_each = self.winner_takes_all(input_saliency_map, avg_length, IMAGE_SIZE_2,IMAGE_SIZE_1,columNum)
                # rgb_gaze_each = self.sample_gaze_from_distri(avg_length, rgb_dis, TOTAL_PCK)
                
                # random_gaze = pd.concat([random_gaze, pd.DataFrame(random_gaze_each)],axis=0)
                center_gaze = pd.concat([center_gaze, pd.DataFrame(center_gaze_each)],axis=0)
                # saliency_gaze = pd.concat([saliency_gaze, pd.DataFrame(saliency_gaze_each)],axis=0)
                # rgb_gaze = pd.concat([rgb_gaze, pd.DataFrame(rgb_gaze_each)],axis=0)
            '''rgb_gaze = pd.concat([rgb_gaze, pd.DataFrame(rgb_gaze_each)],axis=0)
            sim_random.append(self.compute_sim(np.ones(TOTAL_PCK) / TOTAL_PCK,gt,TOTAL_PCK))
            sim_center.append(self.compute_sim(center_dis,gt,TOTAL_PCK))
            sim_saliency.append(self.compute_sim(saliency_dis,gt,TOTAL_PCK))
            sim_rgb.append(self.compute_sim(rgb_dis,gt,TOTAL_PCK)) '''   

        # random_gaze.to_csv(self.saveFolder + 'gaze_random.csv', index=False)
        center_gaze.to_csv(self.saveFolder + 'gaze_center.csv', index=False)
        # saliency_gaze.to_csv(self.saveFolder + 'gaze_saliency.csv', index=False)
        # rgb_gaze.to_csv(self.saveFolder + 'gaze_rgb_similarity.csv', index=False)

        '''print('sim_random:', np.mean(sim_random))
        print('sim_center:', np.mean(sim_center))
        print('sim_saliency:', np.mean(sim_saliency))
        print('sim_rgb:', np.mean(sim_rgb))'''
        

if __name__ == '__main__':
    training_dataset_choice = 'all'
    testing_dataset_choice = 'all'
    saveFolder = './dataset/checkEvaluation/'
    processFolder = './dataset/processdata/'
    logFile = 'res/train_all_test_all_random_time_PE2'
    datapath = './dataset/processdata/dataset_Q123_mousedel_time_new'
    indexFile = './dataset/processdata/splitlist_all_time.txt'
    b = Benchmark(training_dataset_choice, testing_dataset_choice, saveFolder,processFolder, datapath, indexFile)
    b.benchmark()
    e = Evaluation(training_dataset_choice, testing_dataset_choice, saveFolder+logFile, datapath, indexFile)
    e.evaluation()

    