import torch
import numpy as np

file = './dataset/processdata/dataset_Q23_time'
train_num_no = 688

def randsplit(file):
    raw_data = torch.load(file)
    data_length = len(raw_data)
    print(F'len = {data_length}')
    
    list = [i for i in range(data_length)]
    np.random.shuffle(list)
    
    traindata = []
    valdata = []
    for i in range(data_length):
        dataindex = list[i]
        if i < train_num_no:
            train = raw_data[dataindex]
            traindata.append(train)
        if i >= train_num_no:
            val = raw_data[dataindex]
            valdata.append(val)
    print(len(traindata))
    print(len(valdata))
    with open('splitlist_time.txt', 'w') as F:
        F.writelines([str(item)+'\n' for item in list])
        F.close()
    torch.save(traindata, "./dataset/splitdata/dataset_Q23_time_train")
    torch.save(valdata, "./dataset/splitdata/dataset_Q23_time_val")
if __name__ == '__main__':
    randsplit(file)
    
