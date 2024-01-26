import numpy as np
import torch
# generate random integer values
import pandas as pd
import sys
sys.path.append('./src/')
from dataBuilders.data_builder import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import random
from matplotlib.patches import Circle,RegularPolygon
import cv2

import matplotlib.colors as mcolors

class Visualization(object):
    def __init__(self, training_dataset_choice, testing_dataset_choice, evaluation_url, gazeNum=10, plotRange=50, showBenchmark=False):
        datapath = './dataset/processdata/dataset_Q123_mousedel_time_new'
        indexFile = './dataset/processdata/splitlist_all_time.txt'
        self.savePlot = './dataset/visualizationPlot/'
        self.shelfFolder = './dataset/img/Question/'
        self.training_dataset_choice = training_dataset_choice
        self.testing_dataset_choice = testing_dataset_choice
        self.ITERATION = 100
        self.drawPoints = 5
        self.gazeNum = gazeNum
        self.plotRange = plotRange
        self.showBenchmark = showBenchmark

        gaze_max = evaluation_url+'/gaze_max.csv'
        # gaze_expect = evaluation_url+'/gaze_expect.csv'

        if showBenchmark:
            gaze_random = './dataset/checkEvaluation/gaze_random.csv'
            gaze_center = './dataset/checkEvaluation/gaze_center.csv'
            gaze_saliency = './dataset/checkEvaluation/gaze_saliency.csv'
            gaze_rgb = './dataset/checkEvaluation/gaze_rgb_similarity.csv'
        
        self.gaze_max = np.array(pd.read_csv(gaze_max))
        # self.gaze_expect = np.array(pd.read_csv(gaze_expect))

        if showBenchmark:
            self.gaze_random = np.array(pd.read_csv(gaze_random))
            self.gaze_saliency = np.array(pd.read_csv(gaze_saliency))
            self.gaze_rgb = np.array(pd.read_csv(gaze_rgb))
            self.gaze_center = np.array(pd.read_csv(gaze_center))

        raw_data = randsplit(datapath, indexFile, 'Test', testing_dataset_choice, training_dataset_choice)

        self.data_length = len(raw_data)
        print(F'len = {self.data_length}')

        self.id = []
        self.layout_id = []
        self.package_target = []
        self.package_seq = []
        self.target_id = []

        for item in raw_data:
            self.package_seq.append(item['package_seq'])
            self.id.append(item['id'])
            self.package_target.append(item['package_target'])
            self.layout_id.append(item['layout_id'])
            self.target_id.append(item['tgt_id'])

    def random_image_position(self,IMAGE_SIZE_1,IMAGE_SIZE_2, IMAGE_RANGE_X, IMAGE_RANGE_Y):
        center_x = IMAGE_SIZE_2 // 2  
        center_y = IMAGE_SIZE_1 // 2  
        x = random.randint(IMAGE_RANGE_X[0] + center_x - self.plotRange, IMAGE_RANGE_X[1]-center_x + self.plotRange)
        y = random.randint(IMAGE_RANGE_Y[0] + center_y - self.plotRange, IMAGE_RANGE_Y[1]-center_y + self.plotRange)
        
        return (x, y)

    def random_select(self,gaze):
        selected_data = []
        for i in range(self.data_length):
            start_index = i * self.ITERATION
            end_index = start_index + self.ITERATION
            simulation_data = gaze[start_index:end_index]
            selected_indices = np.random.choice(simulation_data.shape[0], size=self.gazeNum, replace=False)
            selected_gaze_trajectories = simulation_data[selected_indices]
            selected_data.append(selected_gaze_trajectories)
        selected_gaze_data = np.concatenate(selected_data, axis=0)
        return selected_gaze_data

    def get_gaze_position(self,package_seq,IMAGE_COLUMN,IMAGE_SIZE_1, IMAGE_SIZE_2):
        package_position_all = []
        for i in range(len(package_seq)):
            package_position = []
            package_element = package_seq[i][~np.isnan(package_seq[i])]
            for j in range(len(package_element)):
                each_package = package_element[j]
                row = each_package // IMAGE_COLUMN
                column = each_package % IMAGE_COLUMN
                IMAGE_RANGE_X = (column * IMAGE_SIZE_2,(column+1) * IMAGE_SIZE_2)
                IMAGE_RANGE_Y = (row * IMAGE_SIZE_1,(row+1) * IMAGE_SIZE_1)
                each_position = self.random_image_position(IMAGE_SIZE_1,IMAGE_SIZE_2, IMAGE_RANGE_X, IMAGE_RANGE_Y)
                package_position.append(each_position)
            package_position_all.append(package_position)
        return package_position_all
        
    def plotGaze(self, shelf, target_pos,gaze_predict, file_name):
        for n in range(len(gaze_predict)):
            if n<10:
                fig, ax = plt.subplots()
                ax.imshow(shelf, alpha =0.75)
                target_x_range, target_y_range = target_pos[0], target_pos[1]
                rect = plt.Rectangle((target_x_range[0], target_y_range[0]), 
                                    target_x_range[1] - target_x_range[0], 
                                    target_y_range[1] - target_y_range[0], 
                                    linewidth=2, edgecolor='green', facecolor='none')
                # for k in range(len(gaze_gt)):
                #     gaze_gt_element = gaze_gt[k]
                #     for i, point in enumerate(gaze_gt_element):
                #         x, y = point[0],point[1]
                #         ax.scatter(x, y, color='green', edgecolors='red')
                #         circle = Circle((x, y), radius=30, edgecolor='black', facecolor=mcolors.TABLEAU_COLORS['tab:red'],alpha = 0.8)
                #         ax.add_artist(circle)
                #         ax.text(x, y, f'{i+1}', ha='center', va='center', fontsize=10, fontweight='bold',color = 'white')
                #         if i > 0:
                #                 ax.arrow(gaze_gt_element[i-1][0], gaze_gt_element[i-1][1], point[0] - gaze_gt_element[i-1][0],
                #                             point[1] - gaze_gt_element[i-1][1], head_width=20, head_length=20, linewidth=1,
                #                             fc=mcolors.TABLEAU_COLORS['tab:red'], ec=mcolors.TABLEAU_COLORS['tab:red'], alpha=1)
                gaze_predict_element = gaze_predict[n]
                for i, point in enumerate(gaze_predict_element):
                    x, y = point[0],point[1]
                    ax.scatter(x, y, color='green', edgecolors='red')
                    circle = Circle((x, y), radius=35, edgecolor='black', facecolor='#E7BE58')
                    ax.add_artist(circle)
                    
                    # triangle = RegularPolygon((x, y), numVertices=3, radius=40,edgecolor='black', facecolor='yellow')
                    # ax.add_patch(triangle)
                    ax.text(x, y, f'{i+1}', ha='center', va='center', fontsize=10, fontweight='bold',color = 'black')
                    if i > 0:
                            ax.arrow(gaze_predict_element[i-1][0], gaze_predict_element[i-1][1], point[0] - gaze_predict_element[i-1][0],
                                        point[1] - gaze_predict_element[i-1][1], head_width=20, head_length=20, linewidth=1,
                                        fc='none', ec='orange', alpha=1)
                ax.add_patch(rect)
                # ax.legend(['Ground Truth', 'Predictions'])
                plt.axis('off')
                # plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
                plt.savefig(self.savePlot + file_name + '_' + str(n) + '.png', bbox_inches='tight', pad_inches=0,dpi=300)
                plt.close()
            


    def forward(self):
        # selected_gaze_predict = self.random_select(self.gaze_expect)
        if self.showBenchmark:
            selected_gaze_random = self.random_select(self.gaze_random)
            selected_gaze_center = self.random_select(self.gaze_center)
            selected_gaze_saliency = self.random_select(self.gaze_saliency)
            selected_gaze_rgb = self.random_select(self.gaze_rgb)
    
        for i in tqdm(range(self.data_length)):
            if self.training_dataset_choice == 'pure':
                if self.testing_dataset_choice == 'wine':
                    TOTAL_PCK = 22
                    IMAGE_SIZE_1 = 449
                    IMAGE_SIZE_2 = 152
                    IMAGE_ROW = 2
                    IMAGE_COLUMN = 11
                elif self.testing_dataset_choice == 'yogurt':
                    TOTAL_PCK = 27
                    IMAGE_SIZE_1 = 305
                    IMAGE_SIZE_2 = 186
                    IMAGE_ROW = 3
                    IMAGE_COLUMN = 9
            elif self.training_dataset_choice == 'all':
                if self.id[i] == 'Q1':
                    TOTAL_PCK = 22
                    IMAGE_SIZE_1 = 449
                    IMAGE_SIZE_2 = 152
                    IMAGE_ROW = 2
                    IMAGE_COLUMN = 11

                elif self.id[i] == 'Q3':
                    TOTAL_PCK = 27
                    IMAGE_SIZE_1 = 305
                    IMAGE_SIZE_2 = 186
                    IMAGE_ROW = 3
                    IMAGE_COLUMN = 9
            
            package_target = self.package_target[i]

            layout_id = self.layout_id[i]
            if self.target_id[i] == 'T3_12' and self.layout_id[i] == 'Q3_10':
                # target_id = self.target_id[i]
                # print('=======')
                # print(i)
                # print(layout_id)
                package_seq = self.package_seq[i]
                IMAGE_RANGE_X_target = (((int(package_target[0]) -1) % IMAGE_COLUMN) * IMAGE_SIZE_2,(((int(package_target[0]) -1) % IMAGE_COLUMN) +1) * IMAGE_SIZE_2)
                IMAGE_RANGE_Y_target = (((int(package_target[0]) -1) // IMAGE_COLUMN) * IMAGE_SIZE_1,(((int(package_target[0]) -1) // IMAGE_COLUMN) +1) * IMAGE_SIZE_1)
                question_img = Image.open(self.shelfFolder + layout_id + '.png')
                question_cropped_range = (0, (1050-IMAGE_SIZE_1*IMAGE_ROW), IMAGE_COLUMN*IMAGE_SIZE_2,  (1050-IMAGE_SIZE_1*IMAGE_ROW)+IMAGE_ROW *IMAGE_SIZE_1)
                question_img_cropped =  question_img.crop(question_cropped_range)
                package_seq = [[x - 1 for x in package_seq]]
                package_position_gt = self.get_gaze_position(np.array(package_seq),IMAGE_COLUMN,IMAGE_SIZE_1, IMAGE_SIZE_2)
                package_position_max = self.get_gaze_position(self.gaze_max[i:(i+1)],IMAGE_COLUMN,IMAGE_SIZE_1, IMAGE_SIZE_2)
                # package_position_muti = self.get_gaze_position(selected_gaze_predict[(i * self.gazeNum):(i * self.gazeNum + self.gazeNum)],IMAGE_COLUMN,IMAGE_SIZE_1, IMAGE_SIZE_2)

                self.plotGaze(question_img_cropped, [IMAGE_RANGE_X_target,IMAGE_RANGE_Y_target],package_position_gt,file_name = './comb2_gazeformer/gt' + str(i))
                self.plotGaze(question_img_cropped, [IMAGE_RANGE_X_target,IMAGE_RANGE_Y_target],package_position_max,file_name = './comb2_gazeformer/best' + str(i))

                # self.plotGaze(question_img_cropped, [IMAGE_RANGE_X_target,IMAGE_RANGE_Y_target], package_position_gt,package_position_muti,file_name = './muti/predict' + str(i))
                
                if self.showBenchmark:
                    package_position_random = self.get_gaze_position(selected_gaze_random[(i * self.gazeNum):(i * self.gazeNum + self.gazeNum)],IMAGE_COLUMN,IMAGE_SIZE_1, IMAGE_SIZE_2)
                    package_position_center = self.get_gaze_position(selected_gaze_center[(i * self.gazeNum):(i * self.gazeNum + self.gazeNum)],IMAGE_COLUMN,IMAGE_SIZE_1, IMAGE_SIZE_2)
                    package_position_saliency = self.get_gaze_position(selected_gaze_saliency[(i * self.gazeNum):(i * self.gazeNum + self.gazeNum)],IMAGE_COLUMN,IMAGE_SIZE_1, IMAGE_SIZE_2)
                    package_position_rgb = self.get_gaze_position(selected_gaze_rgb[(i * self.gazeNum):(i * self.gazeNum + self.gazeNum)],IMAGE_COLUMN,IMAGE_SIZE_1, IMAGE_SIZE_2)
                    # self.plotGaze(question_img_cropped, [IMAGE_RANGE_X_target,IMAGE_RANGE_Y_target],package_position_random,file_name = './random/predict' + str(i))
                    # self.plotGaze(question_img_cropped, [IMAGE_RANGE_X_target,IMAGE_RANGE_Y_target],package_position_center,file_name = './center/' + str(i))
                    # self.plotGaze(question_img_cropped, [IMAGE_RANGE_X_target,IMAGE_RANGE_Y_target],package_position_saliency,file_name = '。/saliency' + str(i))
                    # self.plotGaze(question_img_cropped, [IMAGE_RANGE_X_target,IMAGE_RANGE_Y_target], package_position_gt,package_position_rgb,file_name = 'rgb' + str(i))
                # exit()

if __name__ == '__main__':
    training_dataset_choice = 'all'
    testing_dataset_choice = 'all'
    evaluation_url = './dataset/checkEvaluation/res/gaze_comb/train_all_test_all_random_time_comb2_gaze'
    v = Visualization(training_dataset_choice, testing_dataset_choice,evaluation_url)
    v.forward()