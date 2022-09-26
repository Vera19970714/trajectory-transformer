#import torch
import numpy as np
import pickle

file = './dataset/processdata/dataset_Q23_time'
train_num_no = 688

def randsplit(file):
    with open(file, "rb") as fp:  # Unpickling
        raw_data = pickle.load(fp)
    #raw_data = torch.load(file)
    data_length = len(raw_data)
    print(F'len = {data_length}')
    
    list = np.arange(data_length)
    np.random.shuffle(list)
    train_index = list[:int(data_length*0.9)]
    test_index = list[-(data_length - int(data_length*0.9)):]
    traindata = np.array(raw_data)[train_index.astype(int)]
    valdata = np.array(raw_data)[test_index.astype(int)]

    print(len(traindata))
    print(len(valdata))
    with open('./dataset/processdata/splitlist_time.txt', 'w') as F:
        F.writelines([str(item)+'\n' for item in list])
        F.close()
    with open("./dataset/processdata/dataset_Q23_time_train", "wb") as fp:  # Pickling
        pickle.dump(traindata, fp)
    with open("./dataset/processdata/dataset_Q23_time_val", "wb") as fp:  # Pickling
        pickle.dump(valdata, fp)

if __name__ == '__main__':
    randsplit(file)
    
