import pandas as pd
import imagehash
import PIL.Image as Image
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from collections import Counter

mark_file = '../dataset/click data/Mark.xlsx'
data_file = '../dataset/poor eyetracking data//ResNetl5_time_Q2.xlsx'
click_data = '../dataset/click data/time_constrain/'
out_path = '../dataset/gaze/time_Q2_mousedel.xlsx'

class CalibrationClean(object):
    def __init__(self):
        pass
            
    def read_excel(self, file_name):
        xls = pd.ExcelFile(file_name)
        self.df = pd.read_excel(xls)

    def hist_similar(self, lh, rh):
        assert len(lh) == len(rh)
        hist = sum(1 - (0 if l == r else float(abs(l - r)) / max(l, r)) for l, r in zip(lh, rh)) / len(lh)
        return hist
        
    def get_index(self, lst=None, item=''):
        return [index for (index,value) in enumerate(lst) if value == item]
    
    def calibrationclean(self):
        count = 0
        count_fail =0
        count_correct =0
        count_uncorrect=0
        count_delete = []
        count_percentage = []
        df_new = pd.DataFrame(columns=['ID', 'Sub_ID', 'Task_ID', 'Target', 'Question','T_Package','T', 'Choice', 'Distractor Similarity','Similarity 1', 'Similarity 2', 'Similarity 3', 'Similarity 4', 'Similarity 5', 'Similarity 6', 'Similarity 7', 'Similarity 8', 'Similarity 9', 'Similarity 10', 'Similarity 11', 'Similarity 12', 'Similarity 13', 'Similarity 14', 'Similarity 15', 'Similarity 16', 'Similarity 17', 'Similarity 18', 'Similarity 19', 'Similarity 20', 'Similarity 21', 'Similarity 22', 'Similarity 23', 'Similarity 24', 'Similarity 25', 'Similarity 26', 'Similarity 27', 'Refix 1', 'Refix 2', 'Refix 3', 'Refix 4', 'Refix 5', 'Refix 6', 'Refix 7', 'Refix 8', 'Refix 9', 'Refix 10', 'Refix 11', 'Refix 12', 'Refix 13', 'Refix 14', 'Refix 15', 'Refix 16', 'Refix 17', 'Refix 18', 'Refix 19', 'Refix 20', 'Refix 21', 'Refix 22','Refix 23', 'Refix 24', 'Refix 25', 'Refix 26', 'Refix 27', 'Revisit 1', 'Revisit 2', 'Revisit 3', 'Revisit 4', 'Revisit 5', 'Revisit 6', 'Revisit 7', 'Revisit 8', 'Revisit 9', 'Revisit 10', 'Revisit 11', 'Revisit 12', 'Revisit 13', 'Revisit 14', 'Revisit 15', 'Revisit 16', 'Revisit 17', 'Revisit 18', 'Revisit 19', 'Revisit 20', 'Revisit 21', 'Revisit 22','Revisit 23', 'Revisit 24', 'Revisit 25', 'Revisit 26', 'Revisit 27','Saliency 1', 'Saliency 2', 'Saliency 3', 'Saliency 4', 'Saliency 5', 'Saliency 6', 'Saliency 7', 'Saliency 8', 'Saliency 9', 'Saliency 10', 'Saliency 11', 'Saliency 12', 'Saliency 13', 'Saliency 14', 'Saliency 15', 'Saliency 16', 'Saliency 17', 'Saliency 18', 'Saliency 19', 'Saliency 20', 'Saliency 21', 'Saliency 22', 'Saliency 23', 'Saliency 24', 'Saliency 25', 'Saliency 26', 'Saliency 27'])
        # df_new = pd.DataFrame(columns=['ID', 'Sub_ID', 'Task_ID', 'Target', 'Question', 'T_Package','T', 'Choice', 'Distractor Similarity','Similarity 1', 'Similarity 2', 'Similarity 3', 'Similarity 4', 'Similarity 5', 'Similarity 6', 'Similarity 7', 'Similarity 8', 'Similarity 9', 'Similarity 10', 'Similarity 11', 'Similarity 12', 'Similarity 13', 'Similarity 14', 'Similarity 15', 'Similarity 16', 'Similarity 17', 'Similarity 18', 'Similarity 19', 'Similarity 20', 'Similarity 21', 'Similarity 22',  'Refix 1', 'Refix 2', 'Refix 3', 'Refix 4', 'Refix 5', 'Refix 6', 'Refix 7', 'Refix 8', 'Refix 9', 'Refix 10', 'Refix 11', 'Refix 12', 'Refix 13', 'Refix 14', 'Refix 15', 'Refix 16', 'Refix 17', 'Refix 18', 'Refix 19', 'Refix 20', 'Refix 21', 'Refix 22', 'Revisit 1', 'Revisit 2', 'Revisit 3', 'Revisit 4', 'Revisit 5', 'Revisit 6', 'Revisit 7', 'Revisit 8', 'Revisit 9', 'Revisit 10', 'Revisit 11', 'Revisit 12', 'Revisit 13', 'Revisit 14', 'Revisit 15', 'Revisit 16', 'Revisit 17', 'Revisit 18', 'Revisit 19', 'Revisit 20', 'Revisit 21', 'Revisit 22','Saliency 1', 'Saliency 2', 'Saliency 3', 'Saliency 4', 'Saliency 5', 'Saliency 6', 'Saliency 7', 'Saliency 8', 'Saliency 9', 'Saliency 10', 'Saliency 11', 'Saliency 12', 'Saliency 13', 'Saliency 14', 'Saliency 15', 'Saliency 16', 'Saliency 17', 'Saliency 18', 'Saliency 19', 'Saliency 20', 'Saliency 21', 'Saliency 22','AbDis 1', 'AbDis 2', 'AbDis 3', 'AbDis 4', 'AbDis 5', 'AbDis 6', 'AbDis 7', 'AbDis 8', 'AbDis 9', 'AbDis 10', 'AbDis 11', 'AbDis 12', 'AbDis 13', 'AbDis 14', 'AbDis 15', 'AbDis 16', 'AbDis 17', 'AbDis 18', 'AbDis 19', 'AbDis 20', 'AbDis 21', 'AbDis 22', 'AbVer 1', 'AbVer 2', 'AbVer 3', 'AbVer 4', 'AbVer 5', 'AbVer 6', 'AbVer 7', 'AbVer 8', 'AbVer 9', 'AbVer 10', 'AbVer 11', 'AbVer 12', 'AbVer 13', 'AbVer 14', 'AbVer 15', 'AbVer 16', 'AbVer 17', 'AbVer 18', 'AbVer 19', 'AbVer 20', 'AbVer 21', 'AbVer 22', 'AbHor 1', 'AbHor 2', 'AbHor 3', 'AbHor 4', 'AbHor 5', 'AbHor 6', 'AbHor 7', 'AbHor 8', 'AbHor 9', 'AbHor 10', 'AbHor 11', 'AbHor 12', 'AbHor 13', 'AbHor 14', 'AbHor 15', 'AbHor 16', 'AbHor 17', 'AbHor 18', 'AbHor 19', 'AbHor 20', 'AbHor 21', 'AbHor 22','R 1', 'R 2', 'R 3','R 4', 'R 5', 'R 6', 'R 7', 'R 8', 'R 9', 'R 10', 'R 11', 'R 12','R 13', 'R 14', 'R 15', 'R 16', 'R 17', 'R 18', 'R 19', 'R 20', 'R 21', 'R 22', 'G 1', 'G 2', 'G 3','G 4', 'G 5', 'G 6', 'G 7', 'G 8', 'G 9', 'G 10', 'G 11', 'G 12','G 13', 'G 14', 'G 15', 'G 16', 'G 17', 'G 18', 'G 19', 'G 20', 'G 21', 'G 22', 'B 1', 'B 2', 'B 3','B 4', 'B 5', 'B 6', 'B 7', 'B 8', 'B 9', 'B 10', 'B 11', 'B 12','B 13', 'B 14', 'B 15', 'B 16', 'B 17', 'B 18', 'B 19', 'B 20', 'B 21', 'B 22'])

        mark_df = pd.read_excel(mark_file)
        words = [str(item) for item in list(self.df["Sub_ID"])]
        word_dict_sorted = {}
        data_len = []
        target_item = []
        for line in words:
            line = line.replace(',',' ').replace('\n',' ').lower()
            for word in line.split():
                if word in word_dict_sorted:
                    word_dict_sorted[word] += 1
                else:
                    word_dict_sorted[word] = 1
        
        for key in word_dict_sorted:
            id_num = word_dict_sorted[key]
            df1 = self.df[self.df["Sub_ID"]==int(key)]
            df1.reset_index(drop=True, inplace=True)
            file_name = mark_df[mark_df["Index"]==int(key)].iloc[0].at['File Name']
            current_experiment_data = pd.read_excel(click_data + file_name + '.xlsx')
            task_words = [str(item) for item in list(df1["ID"])]
            word_dict_task ={}
            
            for line in task_words:
                line = line.replace(',',' ').replace('\n',' ').lower()
                for word in line.split():
                    if word in word_dict_task:
                        word_dict_task[word] += 1
                    else:
                        word_dict_task[word] = 1
            for keys in word_dict_task:
                df2 = df1[df1["ID"]==int(keys)]
                df2.reset_index(drop=True, inplace=True)
                question_name = list(df2["Question"])[0]
                target_name = list(df2["Target"])[0]
                package = list(df2["Choice"])
                target_package = list(df2['T_Package'])[0]
                if question_name+'.png' not in list(current_experiment_data["MediaName"]):
                    df_new = pd.concat([df_new, df2])
                    data_len.append(len(package))
                    target_item.append(target_package)
                    count +=1
                    count_fail +=1
                else:
                    click_id=current_experiment_data[current_experiment_data["MediaName"]==question_name+'.png']
                    x_coordinate=click_id.iloc[0].at['MouseEventX (ADCSpx)']
                    y_coordinate=click_id.iloc[0].at['MouseEventY (ADCSpx)']

                    if question_name.startswith('Q1'):
                        if x_coordinate in range(1,153) and y_coordinate in range(153, 602):
                            Package=1
                        elif x_coordinate in range(153,305) and y_coordinate in range(153, 602):
                            Package=2
                        elif x_coordinate in range(305,457) and y_coordinate in range(153, 602):
                            Package=3
                        elif x_coordinate in range(457,610) and y_coordinate in range(153, 602):
                            Package=4
                        elif x_coordinate in range(610,763) and y_coordinate in range(153, 602):
                            Package=5
                        elif x_coordinate in range(763,916) and y_coordinate in range(153, 602):
                            Package=6
                        elif x_coordinate in range(916,1069) and y_coordinate in range(153, 602):
                            Package=7
                        elif x_coordinate in range(1069,1222) and y_coordinate in range(153, 602):
                            Package=8
                        elif x_coordinate in range(1222,1375) and y_coordinate in range(153, 602):
                            Package=9
                        elif x_coordinate in range(1375,1528) and y_coordinate in range(153, 602):
                            Package=10
                        elif x_coordinate in range(1528,1681) and y_coordinate in range(153, 602):
                            Package=11
                        elif x_coordinate in range(1,153) and y_coordinate in range(603, 1051):
                            Package=12
                        elif x_coordinate in range(153,305) and y_coordinate in range(603, 1051):
                            Package=13
                        elif x_coordinate in range(305,457) and y_coordinate in range(603, 1051):
                            Package=14
                        elif x_coordinate in range(457,610) and y_coordinate in range(603, 1051):
                            Package=15
                        elif x_coordinate in range(610,763) and y_coordinate in range(603, 1051):
                            Package=16
                        elif x_coordinate in range(763,916) and y_coordinate in range(603, 1051):
                            Package=17
                        elif x_coordinate in range(916,1069) and y_coordinate in range(603, 1051):
                            Package=18
                        elif x_coordinate in range(1069,1222) and y_coordinate in range(603, 1051):
                            Package=19
                        elif x_coordinate in range(1222,1375) and y_coordinate in range(603, 1051):
                            Package=20
                        elif x_coordinate in range(1375,1528) and y_coordinate in range(603, 1051):
                            Package=21
                        elif x_coordinate in range(1528,1681) and y_coordinate in range(603, 1051):
                            Package=22
                        else:
                            Package=0

                    elif question_name.startswith('Q2'):
                        if x_coordinate in range(1,187) and y_coordinate in range(166, 461):
                            Package =1
                        elif x_coordinate in range(187,373) and y_coordinate in range(166, 461):
                            Package=2
                        elif x_coordinate in range(373,559) and y_coordinate in range(166, 461):
                            Package=3
                        elif x_coordinate in range(559,746) and y_coordinate in range(166, 461):
                            Package=4
                        elif x_coordinate in range(746,933) and y_coordinate in range(166, 461):
                            Package=5
                        elif x_coordinate in range(933,1120) and y_coordinate in range(166, 461):
                            Package=6
                        elif x_coordinate in range(1120,1307) and y_coordinate in range(166, 461):
                            Package=7
                        elif x_coordinate in range(1307,1494) and y_coordinate in range(166, 461):
                            Package=8
                        elif x_coordinate in range(1494,1681) and y_coordinate in range(166, 461):
                            Package=9
                        elif x_coordinate in range(1,187) and y_coordinate in range(461, 756):
                            Package=10
                        elif x_coordinate in range(187,373) and y_coordinate in range(461, 756):
                            Package=11
                        elif x_coordinate in range(373,559) and y_coordinate in range(461, 756):
                            Package=12
                        elif x_coordinate in range(559,746) and y_coordinate in range(461, 756):
                            Package=13
                        elif x_coordinate in range(746,933) and y_coordinate in range(461, 756):
                            Package=14
                        elif x_coordinate in range(933,1120) and y_coordinate in range(461, 756):
                            Package=15
                        elif x_coordinate in range(1120,1307) and y_coordinate in range(461, 756):
                            Package=16
                        elif x_coordinate in range(1307,1494) and y_coordinate in range(461, 756):
                            Package=17
                        elif x_coordinate in range(1494,1681) and y_coordinate in range(461, 756):
                            Package=18
                        elif x_coordinate in range(1,187) and y_coordinate in range(757, 1051):
                            Package=19
                        elif x_coordinate in range(187,373) and y_coordinate in range(757, 1051):
                            Package=20
                        elif x_coordinate in range(373,559) and y_coordinate in range(757, 1051):
                            Package=21
                        elif x_coordinate in range(559,746) and y_coordinate in range(757, 1051):
                            Package=22
                        elif x_coordinate in range(746,933) and y_coordinate in range(757, 1051):
                            Package=23
                        elif x_coordinate in range(933,1120) and y_coordinate in range(757, 1051):
                            Package=24
                        elif x_coordinate in range(1120,1307) and y_coordinate in range(757, 1051):
                            Package=25
                        elif x_coordinate in range(1307,1494) and y_coordinate in range(757, 1051):
                            Package=26
                        elif x_coordinate in range(1494,1681) and y_coordinate in range(757, 1051):
                            Package=27
                        else:
                            Package=0
                    
                    elif question_name.startswith('Q3'):
                        if x_coordinate in range(1,187) and y_coordinate in range(136, 441):
                            Package=1
                        elif x_coordinate in range(187,373) and y_coordinate in range(136, 441):
                            Package=2
                        elif x_coordinate in range(373,559) and y_coordinate in range(136, 441):
                            Package=3
                        elif x_coordinate in range(559,746) and y_coordinate in range(136, 441):
                            Package=4
                        elif x_coordinate in range(746,933) and y_coordinate in range(136, 441):
                            Package=5
                        elif x_coordinate in range(933,1120) and y_coordinate in range(136, 441):
                            Package=6
                        elif x_coordinate in range(1120,1307) and y_coordinate in range(136, 441):
                            Package=7
                        elif x_coordinate in range(1307,1494) and y_coordinate in range(136, 441):
                            Package=8
                        elif x_coordinate in range(1494,1681) and y_coordinate in range(136, 441):
                            Package=9
                        elif x_coordinate in range(1,187) and y_coordinate in range(441, 746):
                            Package=10
                        elif x_coordinate in range(187,373) and y_coordinate in range(441, 746):
                            Package=11
                        elif x_coordinate in range(373,559) and y_coordinate in range(441, 746):
                            Package=12
                        elif x_coordinate in range(559,746) and y_coordinate in range(441, 746):
                            Package=13
                        elif x_coordinate in range(746,933) and y_coordinate in range(441, 746):
                            Package=14
                        elif x_coordinate in range(933,1120) and y_coordinate in range(441, 746):
                            Package=15
                        elif x_coordinate in range(1120,1307) and y_coordinate in range(441, 746):
                            Package=16
                        elif x_coordinate in range(1307,1494) and y_coordinate in range(441, 746):
                            Package=17
                        elif x_coordinate in range(1494,1681) and y_coordinate in range(441, 746):
                            Package=18
                        elif x_coordinate in range(1,187) and y_coordinate in range(746, 1051):
                            Package=19
                        elif x_coordinate in range(187,373) and y_coordinate in range(746, 1051):
                            Package=20
                        elif x_coordinate in range(373,559) and y_coordinate in range(746, 1051):
                            Package=21
                        elif x_coordinate in range(559,746) and y_coordinate in range(746, 1051):
                            Package=22
                        elif x_coordinate in range(746,933) and y_coordinate in range(746, 1051):
                            Package=23
                        elif x_coordinate in range(933,1120) and y_coordinate in range(746, 1051):
                            Package=24
                        elif x_coordinate in range(1120,1307) and y_coordinate in range(746, 1051):
                            Package=25
                        elif x_coordinate in range(1307,1494) and y_coordinate in range(746, 1051):
                            Package=26
                        elif x_coordinate in range(1494,1681) and y_coordinate in range(746, 1051):
                            Package=27
                        else:
                            Package=0
                    
                    if Package in package:
                        count +=1
                        length = len(package)
                        
                        target_item.append(target_package)
                        if int(target_package) == int(Package):
                            count_correct += 1
                        else:
                            count_uncorrect += 1
                        step=0
                        for i in range(length-1,-1,-1):
                            if package[i]==int(Package):
                                break
                            else:
                                step+=1
                                df2.drop(df2.index[i],inplace=True) 
                        data_len.append(df2.shape[0])
                        count_percentage.append(step/length)
                        count_delete.append(step)
                        df_new = pd.concat([df_new, df2])
        df_new.reset_index().drop(['index'],axis=1)
        df_new.to_excel(out_path, index=False)   
        counter = Counter(count_delete)
        counter_perc = Counter(count_percentage)
        lst = list(counter) 
        lst2 = list(counter.values())
        del(lst[0])
        del(lst2[0])
        # plt.plot(lst,lst2, 'ro')
        # plt.xlabel('Mouse Noise Length')
        # plt.ylabel('Frequency')
        # plt.show()

        print('count:',count)   
        print('count_fail:',count_fail)    
        print('count_correct:',count_correct)
        print('count_uncorrect:',count_uncorrect)
        print('count_delete:',counter)
        
            

if __name__ == '__main__':
    CalibrationClean = CalibrationClean()
    CalibrationClean.read_excel(data_file)
    CalibrationClean.calibrationclean()

    

                


        
