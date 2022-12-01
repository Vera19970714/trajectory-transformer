#import torch
#import torchvision.transforms as transforms
from cgi import test
import pandas as pd
import numpy as np
import cv2 as cv
from collections import Counter
import pickle
from tqdm import tqdm

img_dir = './dataset/img/Question/'
data_dir1 = './dataset/gaze/time_Q2_mousedel_similarity.xlsx'
data_dir2 = './dataset/gaze/time_Q3_mousedel_similarity.xlsx'
target_dir = './dataset/img/Target/'


class CUT_PIC(object):
    def __init__(self):
        pass

    def read_excel(self, file_name):
        xls = pd.ExcelFile(file_name)
        self.df = pd.read_excel(xls)

    def cut_pic(self):
        dataset = []
        dataset_similarity = []
        df_data1 = pd.read_excel(data_dir1)
        df_data2 = pd.read_excel(data_dir2)
        words1 = [str(item) for item in list(df_data1["ID"])]
        words2 = [str(item) for item in list(df_data2["ID"])]


        word_dict_sorted1 = Counter(words1)
        word_dict_sorted2 = Counter(words2)

        for i in range(2):
            if i==0:
                dict = word_dict_sorted1
                df_ori = df_data1
            elif i==1:
                dict = word_dict_sorted2
                df_ori = df_data2

            for key in tqdm(dict):
                df1 = df_ori[df_ori["ID"]==int(key)]
                df1.reset_index(drop=True, inplace=True)
                question_name = list(df1["Question"])[0]
                package_seq = list(df1["Choice"])
                package_target = list(df1["T_Package"])[0]
                package_target  = [package_target]  # (np.repeat(package_target,27))  
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
                    dim = (93, 150)


                elif question_name.startswith('Q3'):
                    IMAGE_SIZE_1 = 305
                    IMAGE_SIZE_2 = 186
                    IMAGE_ROW = 3
                    IMAGE_COLUMN = 9
                    CROP_RANGE_1 = 245
                    CROP_RANGE_2 = 162
                    dim = (93, 150)
                
                question_img = cv.imread(img_dir + question_name + '.png')
                question_img = cv.cvtColor(question_img,cv.COLOR_BGR2RGB)


                for y in range(1, IMAGE_ROW + 1):
                    for x in range(1, IMAGE_COLUMN + 1):
                        img_cropped_feature = question_img[((1050-IMAGE_SIZE_1*IMAGE_ROW)+(y-1)*IMAGE_SIZE_1):((1050-IMAGE_SIZE_1*IMAGE_ROW)+y*IMAGE_SIZE_1), ((x-1)*IMAGE_SIZE_2):x*IMAGE_SIZE_2]
                        
                        # img_cropped_feature = img_cropped_feature[int((IMAGE_SIZE_1 - CROP_RANGE_1) / 2) : int(IMAGE_SIZE_1 - (IMAGE_SIZE_1 - CROP_RANGE_1) / 2), int((IMAGE_SIZE_2 - CROP_RANGE_2) / 2) : int(IMAGE_SIZE_2 - (IMAGE_SIZE_2 - CROP_RANGE_2)/2)]
                        img_cropped_feature = cv.resize(img_cropped_feature, dim)
                        img_cropped_feature = cv.normalize(img_cropped_feature, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
                        Question_img_feature.append(img_cropped_feature)
                        
                dataset_dict['package_target'] = package_target
                dataset_dict['package_seq'] =  package_seq
                dataset_dict['question_img_feature'] =  Question_img_feature


                if question_name.startswith('Q2') or question_name.startswith('Q3'):
                    dataset.append(dataset_dict)
            


        with open("./dataset/processdata/dataset_Q23_mousedel_time", "wb") as fp:  # Pickling
            pickle.dump(dataset, fp)

        print("Finish...")

# ==================================== preprocess for baseline model ===================================

class SAVE_PIC(object):
    def __init__(self):
            pass

    def read_excel(self, file_name):
        xls = pd.ExcelFile(file_name)
        self.df = pd.read_excel(xls)

    def save_pic(self):
        dataset1 = []
        dataset2 = []
        dataset3 = []
        words = [str(item) for item in list(self.df["id"])]
        #word_dict_sorted = {}

        word_dict_sorted2 = Counter(words)
        
        for key in tqdm(word_dict_sorted2):
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
            cropped_Question_img_feature = []
            
            
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
                dim = (93, 150)
                dim_whole = (837,450)


            elif question_name.startswith('Q3'):
                IMAGE_SIZE_1 = 305
                IMAGE_SIZE_2 = 186
                IMAGE_ROW = 3
                IMAGE_COLUMN = 9
                CROP_RANGE_1 = 245
                CROP_RANGE_2 = 162
                dim = (93, 150)
                dim_whole = (837,450)
            
            question_img = cv.imread(img_dir + question_name + '.png')
            question_img = cv.cvtColor(question_img,cv.COLOR_BGR2RGB)

            img_feature = question_img[(1050-IMAGE_SIZE_1*IMAGE_ROW):1050, 0:1680]
            img_feature = cv.resize(img_feature, dim_whole)
            img_feature = cv.normalize(img_feature, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
            Question_img_feature.append(img_feature)

            for y in range(1, IMAGE_ROW + 1):
                for x in range(1, IMAGE_COLUMN + 1):
                    img_cropped_feature = question_img[((1050-IMAGE_SIZE_1*IMAGE_ROW)+(y-1)*IMAGE_SIZE_1):((1050-IMAGE_SIZE_1*IMAGE_ROW)+y*IMAGE_SIZE_1), ((x-1)*IMAGE_SIZE_2):x*IMAGE_SIZE_2]
                    
                    # img_cropped_feature = img_cropped_feature[int((IMAGE_SIZE_1 - CROP_RANGE_1) / 2) : int(IMAGE_SIZE_1 - (IMAGE_SIZE_1 - CROP_RANGE_1) / 2), int((IMAGE_SIZE_2 - CROP_RANGE_2) / 2) : int(IMAGE_SIZE_2 - (IMAGE_SIZE_2 - CROP_RANGE_2)/2)]
                    img_cropped_feature = cv.resize(img_cropped_feature, dim)
                    img_cropped_feature = cv.normalize(img_cropped_feature, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
                    cropped_Question_img_feature.append(img_cropped_feature)
                    

            dataset_dict['package_target'] = package_target
            dataset_dict['package_seq'] =  package_seq
            dataset_dict['question_img_feature'] =  Question_img_feature
            dataset_dict['cropped_question_img_feature'] =  cropped_Question_img_feature

            #if question_name.startswith('Q1'):
            #    dataset1.append(dataset_dict)
            if question_name.startswith('Q2') or question_name.startswith('Q3'):
                dataset2.append(dataset_dict)
            # elif question_name.startswith('Q3'):    
            #     dataset3.append(dataset_dict)

        #torch.save(dataset1, "./dataset/processdata/dataset_Q1_time")
        #np.save(dataset2, "./dataset/processdata/dataset_Q23_time")
        # torch.save(dataset3, "./dataset/processdata/dataset_Q3")
        with open("./dataset/processdata/dataset_Q23_baseline_time", "wb") as fp:  # Pickling
            pickle.dump(dataset2, fp)

        print("Finish...")

    

if __name__ == '__main__':
    CUT_PIC = CUT_PIC()
    CUT_PIC.cut_pic()

# if __name__ == '__main__':
#     SAVE_PIC= SAVE_PIC()
#     SAVE_PIC.read_excel(data_dir) 
#     SAVE_PIC.save_pic()



