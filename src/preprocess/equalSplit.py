import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
from itertools import groupby
import pandas as pd


class EqualSplit(object):
    def __init__(self):
        datapath = './dataset/processdata/dataset_Q123_mousedel_time_raw'
        with open(datapath, "rb") as fp: 
            raw_data = pickle.load(fp)

        self.data_length = len(raw_data)
        print(F'len = {self.data_length}')
        self.sub_id = []
        self.id = []
        self.layout_id = []
        self.package_seq = []
        self.target = []
        for item in raw_data:
            self.target.append(item['package_target'])
            self.id.append(item['id'])
            self.layout_id.append(item['layout_id'])
            self.sub_id.append(item['sub_id'])
            sequence_list = [x - 1 for x in item['package_seq']]
            self.package_seq.append(sequence_list)
            
        self.target = [int(self.target[i][0])-1 for i in range(len(self.target))]

    def behavior_utils(self, target, gaze_element):
        search_len = 0
        refix_len = 0
        revisit_len = 0
        previous_visited = []
        for k, g in groupby(gaze_element):
            subiterator_len = len(list(g))
            if k in previous_visited:
                revisit_len += 1
            else:
                search_len += 1
            if subiterator_len > 1:
                refix_len += (subiterator_len - 1)
            previous_visited.append(k)
        assert search_len + refix_len + revisit_len == len(gaze_element)
        return int(target == gaze_element[-1]),len(gaze_element), search_len / len(gaze_element), refix_len / len(gaze_element), revisit_len / len(gaze_element)

    
    def equalSplit(self):
        dataset = []
        number_list = list(range(0, self.data_length))
        names = ['id','sub_id', 'layout_id', 'correct', 'length','search','refix','revisit']
        for i in range(self.data_length):
            res = {name: {} for name in names}
            res['id'] = number_list[i]
            res['sub_id'] = self.sub_id[i]
            res['layout_id'] = self.layout_id[i]
            res['correct'], res['length'], res['search'], res['refix'],res['revisit'] = self.behavior_utils(self.target[i], self.package_seq[i])
            dataset.append(res)

        df = pd.DataFrame(dataset)
        # for group_name, group_data in grouped_data:
        #     plt.hist(group_data['length'], bins=10, alpha=0.5)
        #     plt.xlabel('Length')
        #     plt.ylabel('Frequency')
        #     plt.title(f'Histogram of Length for Layout ID: {group_name}')
        #     plt.savefig('./Image/' + f'histogram_{group_name}.png')
        #     plt.clf()  # Clear the current figure for the next iteration

        grouped_data = df.groupby('layout_id')
        subject_ids = []
        subject_data = []
        for group_name, group_data in grouped_data:
            subject_ids.append(group_name)
            subject_data.append(group_data)

        train_subjects, remaining_subjects, train_data, remaining_data = train_test_split(
            subject_ids, subject_data, train_size=0.8, random_state=44)
        
        valid_subjects, test_subjects, valid_data, test_data = train_test_split(
            remaining_subjects, remaining_data, test_size=0.5, random_state=44)

        train_data = pd.concat(train_data)
        valid_data = pd.concat(valid_data)
        test_data = pd.concat(test_data)

        train_task_ids = train_data['id'].tolist()
        valid_task_ids = valid_data['id'].tolist()
        test_task_ids = test_data['id'].tolist()

        combined_task_ids = train_task_ids + valid_task_ids + test_task_ids

        file_path = './dataset/processdata/splitlist_equal.txt'

        with open(file_path, 'w') as file:
            for task_id in combined_task_ids:
                file.write(str(task_id) + '\n')

        print('Task IDs saved to', file_path)
        
        # for i in range(len(names)-3):
        #     print('Training set distribution:')
        #     print(train_data[names[i+3]].describe())

        #     print('Validation set length distribution:')
        #     print(valid_data[names[i+3]].describe())

        #     print('Test set length distribution:')
        #     print(test_data[names[i+3]].describe())

            
if __name__ == '__main__':

    b = EqualSplit()
    b.equalSplit()