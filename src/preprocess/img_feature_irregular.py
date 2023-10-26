#import torch
#import torchvision.transforms as transforms
from cgi import test
import pandas as pd
import numpy as np
import cv2 as cv
from collections import Counter
import pickle
from tqdm import tqdm

img = './dataset/img/shelf.jpg'
data_dir = './dataset/gaze/irregular.xlsx'


class CUT_PIC_Iregular(object):
    def __init__(self, productshot_choice, file_name):
        self.productshot_choice = productshot_choice  # yes, no, all
        self.file_name = file_name

    def read_excel(self, file_name):
        xls = pd.ExcelFile(file_name)
        self.df = pd.read_excel(xls)

    def crop_img(self, img, y_top, y_bottom, x_left, x_right, dim):
        crop_img = img[y_top:y_bottom, x_left:x_right]
        dim = (int((x_right-x_left)/2), int((y_bottom-y_top)/2))
        crop_img = cv.resize(crop_img, dim)
        coord = np.array([(y_top+(y_bottom-y_top)/2)/140, (x_left+(x_right-x_left)/2)/155])
        self.mapping.append(coord-1)
        #self.count += 1
        print('size: ', crop_img.shape, 'center: ', coord)
        crop_img = cv.normalize(crop_img, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        self.x += dim[0]
        self.y += dim[1]
        return (crop_img, coord)
    
    def cut_pic_irregular(self):
        dataset = []
        df_data = pd.read_excel(data_dir)
        words = [str(item) for item in list(df_data["ID"])]
        word_dict_sorted = Counter(words)
        print('irregular size:', len(word_dict_sorted))
        question_img = cv.imread(img)
        question_img = cv.cvtColor(question_img,cv.COLOR_BGR2RGB)
        package_num = 43
        dim = (84, 224)
        Question_img_feature = []
        self.x = 0
        self.y = 0
        #self.count = 0
        self.mapping = []
        
        for i in range(1,package_num+1):
            if i==1:
                img_cropped_feature = self.crop_img(question_img, 0, 280, 100, 210, dim)
            elif i==2:
                img_cropped_feature = self.crop_img(question_img, 0, 280, 210, 315, dim)
            elif i==3:
                img_cropped_feature = self.crop_img(question_img, 0, 280, 315, 420, dim)
            elif i==4:
                img_cropped_feature = self.crop_img(question_img, 0, 280, 420, 527, dim)
            elif i==5:
                img_cropped_feature = self.crop_img(question_img, 0, 280, 527, 635, dim)
            elif i==6:
                img_cropped_feature = self.crop_img(question_img, 0, 280, 635, 746, dim)
            elif i==7:
                img_cropped_feature = self.crop_img(question_img, 0, 280, 746, 853, dim)
            elif i==8:
                img_cropped_feature = self.crop_img(question_img, 0, 280, 853, 958, dim)
            elif i==9:
                img_cropped_feature = self.crop_img(question_img, 0, 280, 958, 1070, dim)
            elif i==10:
                img_cropped_feature = self.crop_img(question_img, 0, 280, 1070, 1185, dim)
            elif i==11:
                img_cropped_feature = self.crop_img(question_img, 280, 550, 100, 210, dim)
            elif i==12:
                img_cropped_feature = self.crop_img(question_img, 280, 550, 210, 315, dim)
            elif i==13:
                img_cropped_feature = self.crop_img(question_img, 280, 550, 315, 420, dim)
            elif i==14:
                img_cropped_feature = self.crop_img(question_img, 280, 550, 420, 525, dim)
            elif i==15:
                img_cropped_feature = self.crop_img(question_img, 280, 550, 525, 630, dim)
            elif i==16:
                img_cropped_feature = self.crop_img(question_img, 280, 550, 630, 740, dim)
            elif i==17:
                img_cropped_feature = self.crop_img(question_img, 280, 550, 740, 845, dim)
            elif i==18:
                img_cropped_feature = self.crop_img(question_img, 280, 550, 845, 960, dim)
            elif i==19:
                img_cropped_feature = self.crop_img(question_img, 280, 550, 960, 1065, dim)
            elif i==20:
                img_cropped_feature = self.crop_img(question_img, 280, 550, 1065, 1185, dim)
            elif i==21:
                img_cropped_feature = self.crop_img(question_img, 550, 800, 100, 175, dim)
            elif i==22:
                img_cropped_feature = self.crop_img(question_img, 550, 800, 175, 250, dim)
            elif i==23:
                img_cropped_feature = self.crop_img(question_img, 550, 800, 250, 325, dim)
            elif i==24:
                img_cropped_feature = self.crop_img(question_img, 550, 800, 325, 405, dim)
            elif i==25:
                img_cropped_feature = self.crop_img(question_img, 550, 800, 405, 485, dim)
            elif i==26:
                img_cropped_feature = self.crop_img(question_img, 550, 800, 485, 575, dim)
            elif i==27:
                img_cropped_feature = self.crop_img(question_img, 550, 800, 575, 660, dim)
            elif i==28:
                img_cropped_feature = self.crop_img(question_img, 550, 800, 660, 750, dim)
            elif i==29:
                img_cropped_feature = self.crop_img(question_img, 550, 800, 750, 825, dim)
            elif i==30:
                img_cropped_feature = self.crop_img(question_img, 550, 800, 825, 915, dim)
            elif i==31:
                img_cropped_feature = self.crop_img(question_img, 550, 800, 915, 1000, dim)
            elif i==32:
                img_cropped_feature = self.crop_img(question_img, 550, 800, 1000, 1087, dim)
            elif i==33:
                img_cropped_feature = self.crop_img(question_img, 550, 800, 1087, 1185, dim)
            elif i==34:
                img_cropped_feature = self.crop_img(question_img, 800, 1024, 100, 195, dim)
            elif i==35:
                img_cropped_feature = self.crop_img(question_img, 800, 1024, 195, 305, dim)
            elif i==36:
                img_cropped_feature = self.crop_img(question_img, 800, 1024, 305, 415, dim)
            elif i==37:
                img_cropped_feature = self.crop_img(question_img, 800, 1024, 415, 530, dim)
            elif i==38:
                img_cropped_feature = self.crop_img(question_img, 800, 1024, 530, 640, dim)
            elif i==39:
                img_cropped_feature = self.crop_img(question_img, 800, 1024, 640, 760, dim)
            elif i==40:
                img_cropped_feature = self.crop_img(question_img, 800, 1024, 760, 875, dim)
            elif i==41:
                img_cropped_feature = self.crop_img(question_img, 800, 1024, 875, 970, dim)
            elif i==42:
                img_cropped_feature = self.crop_img(question_img, 800, 1024, 970, 1080, dim)
            elif i==43:
                img_cropped_feature = self.crop_img(question_img, 800, 1024, 1080, 1185, dim)
            Question_img_feature.append(img_cropped_feature)

        print('avg dim: ', self.x/43, self.y/43)
        for key in tqdm(word_dict_sorted):
            dataset_dict = {}
            df1 = df_data[df_data["ID"]==int(key)]
            df1.reset_index(drop=True, inplace=True)
            package_seq = list(df1["Choice"])
            package_target = list(df1["T_Package"])[0]
            package_target  = [package_target]
            condition =  list(df1["Condition"])[0]
            if self.productshot_choice == 'no':
                if condition == 0:
                    dataset_dict['package_target'] = package_target
                    dataset_dict['package_seq'] =  package_seq
                    dataset_dict['question_img_feature'] =  Question_img_feature
            
            if self.productshot_choice == 'yes':
                if condition == 1:
                    dataset_dict['package_target'] = package_target
                    dataset_dict['package_seq'] =  package_seq
                    dataset_dict['question_img_feature'] =  Question_img_feature
            
            if self.productshot_choice == 'all':
                dataset_dict['package_target'] = package_target
                dataset_dict['package_seq'] =  package_seq
                dataset_dict['question_img_feature'] =  Question_img_feature

            dataset.append(dataset_dict)

        with open(self.file_name, "wb") as fp:  
            pickle.dump(dataset, fp)

        print(self.mapping)
        self.mapping.append(np.array([2, 5]))
        with open("./dataset/processdata/mapping", "wb") as fp:
            pickle.dump(self.mapping, fp)
        print("Finish...")


    

if __name__ == '__main__':
    CUT_PIC = CUT_PIC_Iregular('yes', "./dataset/processdata/dataset_irregular_yes")
    CUT_PIC.cut_pic_irregular()




