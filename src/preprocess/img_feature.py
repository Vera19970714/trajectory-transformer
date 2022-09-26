#import torch
#import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import cv2 as cv
from collections import Counter
import pickle
img_dir = './dataset/img/Question/'
data_dir = './dataset/gaze/whole_target_correct_time.xlsx'
target_dir = './dataset/img/Target/'


class CUT_PIC(object):
    def __init__(self):
            pass

    def read_excel(self, file_name):
            xls = pd.ExcelFile(file_name)
            self.df = pd.read_excel(xls)

    def cut_pic(self):
        dataset1 = []
        dataset2 = []
        dataset3 = []
        words = [str(item) for item in list(self.df["id"])]
        #word_dict_sorted = {}

        word_dict_sorted2 = Counter(words)
        
        for key in word_dict_sorted2:
            #id_num = word_dict_sorted[key]
            df1 = self.df[self.df["id"]==int(key)]
            df1.reset_index(drop=True, inplace=True)
            question_name = list(df1["question"])[0]
            package_seq = list(df1["package"])
            package_target = list(df1["target_package"])[0]
            package_target  = [package_target]  # (np.repeat(package_target,27))
            #loader = transforms.ToTensor()
            
            dataset_dict = {}
            Question_img_feature = []
            
            
            if question_name.startswith('Q1'):
                continue
                IMAGE_SIZE_1 = 449
                IMAGE_SIZE_2 = 152
                IMAGE_ROW = 2
                IMAGE_COLUMN = 11
                CROP_RANGE_1 = 389
                CROP_RANGE_2 = 106 
                dim = (106, 390)
 

            elif question_name.startswith('Q2'):
                IMAGE_SIZE_1 = 295
                IMAGE_SIZE_2 = 186
                IMAGE_ROW = 3
                IMAGE_COLUMN = 9
                CROP_RANGE_1 = 239
                CROP_RANGE_2 = 116
                dim = (186, 300)


            elif question_name.startswith('Q3'):
                IMAGE_SIZE_1 = 305
                IMAGE_SIZE_2 = 186
                IMAGE_ROW = 3
                IMAGE_COLUMN = 9
                CROP_RANGE_1 = 245
                CROP_RANGE_2 = 162
                dim = (186, 300)
            
            question_img = cv.imread(img_dir + question_name + '.png')
            question_img = cv.cvtColor(question_img,cv.COLOR_BGR2RGB)
            
            for y in range(1, IMAGE_ROW + 1):
                for x in range(1, IMAGE_COLUMN + 1):
                    img_cropped_feature = question_img[((1050-IMAGE_SIZE_1*IMAGE_ROW)+(y-1)*IMAGE_SIZE_1):((1050-IMAGE_SIZE_1*IMAGE_ROW)+y*IMAGE_SIZE_1), ((x-1)*IMAGE_SIZE_2):x*IMAGE_SIZE_2]
                    
                    # img_cropped_feature = img_cropped_feature[int((IMAGE_SIZE_1 - CROP_RANGE_1) / 2) : int(IMAGE_SIZE_1 - (IMAGE_SIZE_1 - CROP_RANGE_1) / 2), int((IMAGE_SIZE_2 - CROP_RANGE_2) / 2) : int(IMAGE_SIZE_2 - (IMAGE_SIZE_2 - CROP_RANGE_2)/2)]
                    img_cropped_feature = cv.resize(img_cropped_feature, dim)
                    #img_cropped_feature = loader(img_cropped_feature).unsqueeze(0)
                    Question_img_feature.append(img_cropped_feature)
                    
            dataset_dict['package_target'] = package_target
            dataset_dict['package_seq'] =  package_seq
            dataset_dict['question_img_feature'] =  Question_img_feature

            #if question_name.startswith('Q1'):
            #    dataset1.append(dataset_dict)
            if question_name.startswith('Q2') or question_name.startswith('Q3'):
                dataset2.append(dataset_dict)
            # elif question_name.startswith('Q3'):    
            #     dataset3.append(dataset_dict)

        #torch.save(dataset1, "./dataset/processdata/dataset_Q1_time")
        #np.save(dataset2, "./dataset/processdata/dataset_Q23_time")
        # torch.save(dataset3, "./dataset/processdata/dataset_Q3")
        with open("./dataset/processdata/dataset_Q23_time", "wb") as fp:  # Pickling
            pickle.dump(dataset2, fp)

        print("Finish...")

if __name__ == '__main__':
    CUT_PIC = CUT_PIC()
    CUT_PIC.read_excel(data_dir) 
    CUT_PIC.cut_pic()


