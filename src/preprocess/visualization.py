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
from matplotlib.patches import Circle


class Visualization(object):
    def __init__(self, training_dataset_choice, testing_dataset_choice, evaluation_url, gazeNum=1, plotRange=50, showBenchmark=False):
        datapath = './dataset/processdata/dataset_Q123_mousedel_time_raw'
        indexFile = './dataset/processdata/splitlist_all.txt'
        self.savePlot = './dataset/visualizationPlot/'
        self.shelfFolder = './dataset/img/Question/'
        self.training_dataset_choice = training_dataset_choice
        self.testing_dataset_choice = testing_dataset_choice
        self.ITERATION = 100
        self.gazeNum = gazeNum
        self.plotRange = plotRange

        gaze_max = evaluation_url+'/gaze_max.csv'
        gaze_expect = evaluation_url+'/gaze_expect.csv'

        if showBenchmark:
            gaze_random = './dataset/checkEvaluation/gaze_random.csv'
            gaze_center = './dataset/checkEvaluation/gaze_center.csv'
            gaze_saliency = './dataset/checkEvaluation/gaze_saliency.csv'
            gaze_rgb = './dataset/checkEvaluation/gaze_rgb_similarity.csv'
        
        self.gaze_max = np.array(pd.read_csv(gaze_max))
        self.gaze_expect = np.array(pd.read_csv(gaze_expect))

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

        for item in raw_data:
            self.package_seq.append(item['package_seq'])
            self.id.append(item['id'])
            self.package_target.append(item['package_target'])
            self.layout_id.append(item['layout_id'])

    def random_image_position(self,IMAGE_SIZE_1,IMAGE_SIZE_2, IMAGE_RANGE_X, IMAGE_RANGE_Y):
        center_x = IMAGE_SIZE_2 // 2  
        center_y = IMAGE_SIZE_1 // 2  
        x = random.randint(IMAGE_RANGE_X[0] + center_x - self.plotRange, IMAGE_RANGE_X[1]-center_x + self.plotRange)
        y = random.randint(IMAGE_RANGE_Y[0] + center_y - self.plotRange, IMAGE_RANGE_Y[1]-center_y + self.plotRange)
        
        return (x, y)


    def plotGaze(self, shelf, target_pos, gaze_pos,file_name):
        save_file = file_name + '.png'
        fig, ax = plt.subplots()
        ax.imshow(shelf)
        ax.set_xlim(0, shelf.size[0])
        ax.set_ylim(shelf.size[1], 0)
        
        for i, point in enumerate(gaze_pos):
            x, y = point[0],point[1]
            ax.scatter(x, y, color='green', edgecolors='red')
            circle = Circle((x, y), radius=40, edgecolor='black', facecolor='yellow')
            ax.add_artist(circle)
            ax.text(x, y, f'{i+1}', ha='center', va='center', fontsize=12, fontweight='bold')

        
        for i in range(len(gaze_pos) - 1):
            x1, y1 = gaze_pos[i]
            x2, y2 = gaze_pos[i + 1]
            dx = x2 - x1
            dy = y2 - y1
            ax.arrow(x1, y1, dx, dy, head_width=20, head_length=20, fc='red', ec='red')
       
        target_x_range, target_y_range = target_pos[0], target_pos[1]
        rect = plt.Rectangle((target_x_range[0], target_y_range[0]), 
                            target_x_range[1] - target_x_range[0], 
                            target_y_range[1] - target_y_range[0], 
                            linewidth=2, edgecolor='green', facecolor='none')
        ax.add_patch(rect)

        plt.savefig(self.savePlot + save_file)


    def forward(self):
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
            elif self.training_dataset_choice == 'mixed':
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
            package_seq = self.package_seq[i]

            question_img = Image.open(self.shelfFolder + layout_id + '.png')
            question_cropped_range = (0, (1050-IMAGE_SIZE_1*IMAGE_ROW), IMAGE_COLUMN*IMAGE_SIZE_2,  (1050-IMAGE_SIZE_1*IMAGE_ROW)+IMAGE_ROW *IMAGE_SIZE_1)
            question_img_cropped =  question_img.crop(question_cropped_range)
            package_seq = [x - 1 for x in package_seq]
            package_position = []
            for j in range(len(package_seq)):
                each_package = package_seq[j]
                row = each_package // IMAGE_COLUMN
                column = each_package % IMAGE_COLUMN
                IMAGE_RANGE_X = (column * IMAGE_SIZE_2,(column+1) * IMAGE_SIZE_2)
                IMAGE_RANGE_Y = (row * IMAGE_SIZE_1,(row+1) * IMAGE_SIZE_1)
                each_position = self.random_image_position(IMAGE_SIZE_1,IMAGE_SIZE_2, IMAGE_RANGE_X, IMAGE_RANGE_Y)
                package_position.append(each_position)
            IMAGE_RANGE_X_target = (((int(package_target[0]) -1) % IMAGE_COLUMN) * IMAGE_SIZE_2,(((int(package_target[0]) -1) % IMAGE_COLUMN) +1) * IMAGE_SIZE_2)
            IMAGE_RANGE_Y_target = (((int(package_target[0]) -1) // IMAGE_COLUMN) * IMAGE_SIZE_1,(((int(package_target[0]) -1) // IMAGE_COLUMN) +1) * IMAGE_SIZE_1)
            self.plotGaze(question_img_cropped, [IMAGE_RANGE_X_target,IMAGE_RANGE_Y_target], package_position,file_name = str(i))

            exit()

if __name__ == '__main__':
    training_dataset_choice = 'mixed'
    testing_dataset_choice = 'wine'
    evaluation_url = './dataset/checkEvaluation/mixed_pe_exp1_alpha9'
    v = Visualization(training_dataset_choice, testing_dataset_choice,evaluation_url)
    v.forward()