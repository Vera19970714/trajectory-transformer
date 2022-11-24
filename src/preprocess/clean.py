import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm


data_dir = './dataset/gaze/whole_target_correct_time.xlsx'
output_dir = './dataset/gaze/whole_target_correct_clean_time.xlsx'


class Clean(object):
    def __init__(self):
            pass

    def read_excel(self, file_name):
            xls = pd.ExcelFile(file_name)
            self.df = pd.read_excel(xls)

    def clean(self):
        words = [str(item) for item in list(self.df["id"])]
        word_dict_sorted2 = Counter(words)
        df_output = pd.DataFrame(columns=['id','sub_num','task_id','duration','X','Y','deltaX','deltaY','package','behavior','target','question','target_package'])
        for key in tqdm(word_dict_sorted2):
            df1 = self.df[self.df["id"]==int(key)]
            df1.reset_index(drop=True, inplace=True)
            package_seq = list(df1["package"])
            package_target = list(df1["target_package"])[0]
            length = len(package_seq)
            for i in range(length-1,-1,-1):
                if package_seq[i]==package_target:
                    break
                else:
                    df1.drop(df1.index[i],inplace=True)   
            df_output = pd.concat([df_output, df1],axis=0)
        df_output.to_excel(output_dir, index=False)
            
            

if __name__ == '__main__':
    Clean= Clean()
    Clean.read_excel(data_dir) 
    Clean.clean()



