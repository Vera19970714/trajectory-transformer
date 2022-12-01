from cgi import test
import pandas as pd
import numpy as np
import cv2 as cv
from collections import Counter
import pickle
from tqdm import tqdm

data_dir1 = './dataset/similaritydata/time_Q2_mousedel_similarity.xlsx'
data_dir2 = './dataset/similaritydata/time_Q3_mousedel_similarity.xlsx'

class Similarity(object):
    def __init__(self):
            pass

    def similarity(self):
        dataset1 = []
        dataset2 = []
        dataset3 = []
        df1 = pd.read_excel(data_dir1)
        df2 = pd.read_excel(data_dir2)
        words1 = [str(item) for item in list(df1["ID"])]
        words2 = [str(item) for item in list(df2["ID"])]

        word_dict_sorted1 = Counter(words1)
        word_dict_sorted2 = Counter(words2)
        
        for key in tqdm(word_dict_sorted1):
            df_part = df1[df1["ID"]==int(key)]
            df_part.reset_index(drop=True, inplace=True)
            question_name = list(df_part["Question"])[0]
            package_seq = list(df_part["Choice"])
            package_similarity = list(df_part.iloc[0,10:37])
            package_saliency  = list(df_part.iloc[0,37:64])
            package_rgb  = list(df_part.iloc[0,64:91])
            
            dataset_dict = {}          
            dataset_dict['package_seq'] = package_seq
            dataset_dict['package_similarity'] =  package_similarity
            dataset_dict['package_saliency'] =  package_saliency
            dataset_dict['package_rgb'] =  package_rgb
            

            dataset2.append(dataset_dict)
        
        for key in tqdm(word_dict_sorted2):
            df_part = df2[df2["ID"]==int(key)]
            df_part.reset_index(drop=True, inplace=True)
            question_name = list(df_part["Question"])[0]
            package_seq = list(df_part["Choice"])
            package_similarity = list(df_part.iloc[0,9:36])
            package_saliency  = list(df_part.iloc[0,36:63])
            package_rgb  = list(df_part.iloc[0,63:91])

            dataset_dict = {}          
            dataset_dict['package_seq'] = package_seq
            dataset_dict['package_similarity'] =  package_similarity
            dataset_dict['package_saliency'] =  package_saliency
            dataset_dict['package_rgb'] =  package_rgb

            dataset2.append(dataset_dict)

       
        with open("./dataset/processdata/dataset_Q23_similarity_mousedel_time", "wb") as fp:  # Pickling
            pickle.dump(dataset2, fp)

        print("Finish...")


if __name__ == '__main__':
    Similarity= Similarity()
    Similarity.similarity()