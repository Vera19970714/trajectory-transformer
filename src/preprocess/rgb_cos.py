from PIL import Image
from tqdm import tqdm
import os
import cv2
import pandas as pd
from pandas import DataFrame
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
from sklearn.preprocessing import StandardScaler
import pandas as pd


img_dir = '../20211101_heatmap/New_Question/'
data_dir = '../20221026_dataclean/calibration clean/mouse delete/time constrain/time_Q3_mousedel.xlsx'
target_dir = '../20211101_heatmap/Target/'
outdir = '../20221026_dataclean/calibration clean/mouse delete/time constrain/time_Q3_mousedel_rgb.xlsx'

class RGB_COS(object):
    def __init__(self):
            pass
    def read_excel(self, file_name):
            xls = pd.ExcelFile(file_name)
            self.df = pd.read_excel(xls)

    def add_margin(self, pil_img, top, right, bottom, left, color):
        width, height = pil_img.size
        new_width = width + right + left
        new_height = height + top + bottom
        result = Image.new(pil_img.mode, (new_width, new_height), color)
        result.paste(pil_img, (left, top))
        return result

    def image_loader(self, image):
        image = image.convert('RGB')
        image = image.resize((186,300))
        loader = np.array(image) / 255
        return loader
        
    def rgb_cos(self):
        whole_length = self.df.shape[0]
        whole_data = pd.DataFrame(columns=['RGB1','RBG2','RGB3','RGB4','RGB5','RGB6','RGB7','RGB8','RGB9','RGB10','RGB11','RGB12','RGB13','RGB14','RGB15','RGB16','RGB17','RGB18','RGB19','RGB20','RGB21','RGB22','RGB23','RGB24','RGB25','RGB26','RGB27'])
        words = [str(item) for item in list(self.df["ID"])]
        word_dict_sorted = {}
        for line in words:
            line = line.replace(',',' ').replace('\n',' ').lower()
            for word in line.split():
                if word in word_dict_sorted:
                    word_dict_sorted[word] += 1
                else:
                    word_dict_sorted[word] = 1
        
        for key in tqdm(word_dict_sorted):
            id_num = word_dict_sorted[key]
            df1 = self.df[self.df["ID"]==int(key)]
            df1.reset_index(drop=True, inplace=True)
            df_length = df1.shape[0]
            question_name = list(df1["Question"])[0]
            package_target = list(df1["T_Package"])[0]
            question_img = Image.open(img_dir + question_name + '.png')

            Question_img_feature = []
            Question_img_meanRBG =[]
            if question_name.startswith('Q1'):
                IMAGE_SIZE_1 = 449
                IMAGE_SIZE_2 = 152
                IMAGE_ROW = 2
                IMAGE_COLUMN = 11  

            elif question_name.startswith('Q2'):
                IMAGE_SIZE_1 = 295
                IMAGE_SIZE_2 = 186
                IMAGE_ROW = 3
                IMAGE_COLUMN = 9

            elif question_name.startswith('Q3'):
                IMAGE_SIZE_1 = 305
                IMAGE_SIZE_2 = 186
                IMAGE_ROW = 3
                IMAGE_COLUMN = 9
            if question_name.startswith('Q3'):
                position_y = int(package_target / IMAGE_COLUMN)
                position_x = package_target - position_y*IMAGE_COLUMN
                question_cropped_range = (0, (1050-IMAGE_SIZE_1*IMAGE_ROW), IMAGE_COLUMN*IMAGE_SIZE_2,  (1050-IMAGE_SIZE_1*IMAGE_ROW)+IMAGE_ROW *IMAGE_SIZE_1)
                question_img_cropped =  question_img.crop(question_cropped_range)
                for y in range(1, IMAGE_ROW + 1):
                    for x in range(1, IMAGE_COLUMN + 1):
                        cropped_range = ((x-1)*IMAGE_SIZE_2, (1050-IMAGE_SIZE_1*IMAGE_ROW)+(y-1)*IMAGE_SIZE_1, x*IMAGE_SIZE_2,  (1050-IMAGE_SIZE_1*IMAGE_ROW)+y*IMAGE_SIZE_1)
                        img_cropped = question_img.crop(cropped_range)
                        img_cropped_feature = self.image_loader(img_cropped)
                        img_cropped_meanR = np.mean(img_cropped_feature[:,:,0])
                        img_cropped_meanG = np.mean(img_cropped_feature[:,:,1])
                        img_cropped_meanB = np.mean(img_cropped_feature[:,:,2])
                        img_cropped_meanRGB = [img_cropped_meanR, img_cropped_meanG, img_cropped_meanB]
                        img_cropped_meanRGB = np.array(img_cropped_meanRGB)
                        
                        Question_img_feature.append(img_cropped_feature)
                        Question_img_meanRBG.append(img_cropped_meanRGB)
                        
                length = len(Question_img_feature)
                target_img_feature = Question_img_feature[int(package_target)-1]
                target_img_meanRGB = Question_img_meanRBG[int(package_target)-1]
                Result_Feature = []
                Result_MeanRGB = []
                for i in range(length):
                    current_img_feature = Question_img_feature[i]
                    current_img_meanRGB = Question_img_meanRBG[i]
                    result_featuretotal = cosine_similarity(target_img_feature.reshape(1,-1),current_img_feature.reshape(1,-1))
                    result_meanRGB = cosine_similarity(target_img_meanRGB.reshape(1,-1),current_img_meanRGB.reshape(1,-1))
                    Result_Feature.append(result_featuretotal)
                ndarray = np.array(Result_Feature).reshape(1,27)
                ndarray = np.tile(ndarray, (df_length,1)) 
                df_part = pd.DataFrame(ndarray)
                df_part.columns=['RGB1','RBG2','RGB3','RGB4','RGB5','RGB6','RGB7','RGB8','RGB9','RGB10','RGB11','RGB12','RGB13','RGB14','RGB15','RGB16','RGB17','RGB18','RGB19','RGB20','RGB21','RGB22','RGB23','RGB24','RGB25','RGB26','RGB27']
                whole_data = pd.concat([whole_data, df_part])
        # whole_data = pd.concat([self.df, whole_data],axis=1)
        whole_data.reset_index().drop(['index'],axis=1)
        whole_data.to_excel(outdir, index=False)        

                    
                    
                
    
               
                
                


if __name__ == '__main__':
    RGB_COS = RGB_COS()
    RGB_COS.read_excel(data_dir) 
    RGB_COS.rgb_cos()


