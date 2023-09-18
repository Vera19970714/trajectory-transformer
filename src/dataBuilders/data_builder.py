import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pytorch_lightning as pl
import pickle
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from numpy import random

def randsplit(file, indexFile, isTrain, cross_dataset):
    with open(file, "rb") as fp:
        raw_data = pickle.load(fp)
    data_length = len(raw_data)

    with open(indexFile) as f:
        lines = f.readlines()
    linesInt = [int(x) for x in lines]

    if cross_dataset == 'None':
        split_num = int(data_length*0.9)
    elif cross_dataset == 'No':
        split_num = 453

    if isTrain == 'Train':
        train_index = np.array(linesInt[:split_num])
        traindata = np.array(raw_data)[train_index.astype(int)]
        return traindata
    elif isTrain == 'Valid':
        test_index = np.array(linesInt[-(data_length - split_num):])
        valdata = np.array(raw_data)[test_index.astype(int)]
        return valdata
    elif isTrain == 'Test':
        test_index = np.array(linesInt[-(data_length - split_num):])
        testdata = np.array(raw_data)[test_index.astype(int)]
        return testdata


def cross_data_split(file, isTrain):
    with open(file, "rb") as fp:
        raw_data = pickle.load(fp)
    shampoo_task = []
    yogurt_task = []
    for index in range(len(raw_data)):
        if raw_data[index]['id'] == 'Q2':
            shampoo_task.append(index)
        elif raw_data[index]['id'] == 'Q3':
            yogurt_task.append(index)
    if isTrain:
        train_index = np.array(shampoo_task)
        traindata = np.array(raw_data)[train_index.astype(int)]
        return traindata
    else:
        val_index = np.array(yogurt_task)
        valdata = np.array(raw_data)[val_index.astype(int)]
        return valdata

'''save_indices_file_code:
    testing_indexes = random.choice(shampoo_task, size=45, replace=False)
    with open("../dataset/processdata/splitlist_shampoo_testing_indices.txt", 'w') as F:
        F.writelines([str(item).replace(' ', '\t') + '\n' for item in testing_indexes])
        F.close()

    training_list = []
    for x in shampoo_task:
        if x not in testing_indexes:
            training_list.append(x)

    with open("../dataset/processdata/splitlist_pure_indices.txt", 'w') as F:
        F.writelines([str(item).replace(' ', '\t') + '\n' for item in training_list])
        F.close()

    sham = random.choice(training_list, size=204, replace=False)
    yog = random.choice(yogurt_task, size=204, replace=False)
    new_list = list(np.concatenate((sham, yog)))
    with open("../dataset/processdata/splitlist_mixed_indices.txt", 'w') as F:
        F.writelines([str(item).replace(' ', '\t') + '\n' for item in new_list])
        F.close()

    yog = list(random.choice(yogurt_task, size=408, replace=False))
    with open("../dataset/processdata/splitlist_cross_indices.txt", 'w') as F:
        F.writelines([str(item).replace(' ', '\t') + '\n' for item in yog])
        F.close()'''

def cross_data_split2(file, isTrain, indexFolder, crossChoice, testing_dataset_choice):
    with open(file, "rb") as fp:
        raw_data = pickle.load(fp)

    if isTrain == 'Train':
        if crossChoice == 'Pure':
            with open(indexFolder + 'splitlist_' + testing_dataset_choice + '_pure_indices.txt') as f:
                lines = f.readlines()
            train_index = np.array([int(x[:-1]) for x in lines])
        elif crossChoice == 'Mixed':
            with open(indexFolder + 'splitlist_' + testing_dataset_choice + '_mixed_indices.txt') as f:
                lines = f.readlines()
            train_index = np.array([int(x[:-1]) for x in lines])
        elif crossChoice == 'Cross':
            with open(indexFolder + 'splitlist_' + testing_dataset_choice + '_cross_indices.txt') as f:
                lines = f.readlines()
            train_index = np.array([int(x[:-1]) for x in lines])
        elif crossChoice == 'Combine':
            with open(indexFolder + 'splitlist_' + testing_dataset_choice + '_combine_indices.txt') as f:
                lines = f.readlines()
            train_index = np.array([int(x[:-1]) for x in lines])
        traindata = np.array(raw_data)[train_index]
        return traindata
    elif isTrain == 'Valid':
        with open(indexFolder + 'splitlist_' + testing_dataset_choice + '_testing_indices.txt') as f:
            lines = f.readlines()
        val_index = np.array([int(x[:-1]) for x in lines])
        valdata = np.array(raw_data)[val_index]
        return valdata
    elif isTrain == 'Test':
        with open(indexFolder + 'splitlist_' + testing_dataset_choice + '_testing_indices.txt') as f:
            lines = f.readlines()
        test_index = np.array([int(x[:-1]) for x in lines])
        testdata = np.array(raw_data)[test_index]
        return testdata

def cross_data_split3(file, isTrain, indexFolder, crossChoice, testing_dataset_choice):
    with open(file, "rb") as fp:
        raw_data = pickle.load(fp)

    if isTrain == 'Train':
        if crossChoice == 'Only':
            with open(indexFolder + 'splitlist_' + testing_dataset_choice + '_only_indices_train.txt') as f:
                lines = f.readlines()
            train_index = np.array([int(x[:-1]) for x in lines])
        traindata = np.array(raw_data)[train_index]
        return traindata
    elif isTrain == 'Valid':
        with open(indexFolder + 'splitlist_' + testing_dataset_choice + '_only_indices_valid.txt') as f:
            lines = f.readlines()
        val_index = np.array([int(x[:-1]) for x in lines])
        valdata = np.array(raw_data)[val_index]
        return valdata
    elif isTrain == 'Test':
        with open(indexFolder + 'splitlist_' + testing_dataset_choice + '_only_indices_test.txt') as f:
            lines = f.readlines()
        test_index = np.array([int(x[:-1]) for x in lines])
        testdata = np.array(raw_data)[test_index]
        return testdata


class FixDataset(Dataset):
    def __init__(self, args, isTrain):
        new_datapath = args.data_path
        indexFile = args.index_folder + 'splitlist_time_mousedel.txt'
        cross_dataset = args.cross_dataset
        testing_dataset_choice = args.testing_dataset_choice
        '''if cross_dataset == 'None' or cross_dataset == 'No':
            raw_data = randsplit(new_datapath, indexFile, isTrain, cross_dataset)
        elif cross_dataset == 'Yes':
            raw_data = cross_data_split(new_datapath, isTrain)
        else:
            print('cross_dataset value ERROR')
            quit()'''
        assert cross_dataset in ['None', 'Pure', 'Mixed', 'Cross', 'Combine', 'Only']
        assert testing_dataset_choice in ['yogurt', 'shampoo']
        if cross_dataset == 'None':
            raw_data = randsplit(new_datapath, indexFile, isTrain, cross_dataset)
        elif cross_dataset == 'Only':
            raw_data = cross_data_split3(new_datapath, isTrain, args.index_folder, cross_dataset, testing_dataset_choice)
        else:
            raw_data = cross_data_split2(new_datapath, isTrain, args.index_folder, cross_dataset, testing_dataset_choice)

        self.data_length = len(raw_data)
        print(F'len = {self.data_length}')
        self.package_target = []
        self.question_img_feature = []
        self.package_sequence = []
        self.args = args

        #i=0
        for item in raw_data:
            self.package_target.append(item['package_target'])
            self.question_img_feature.append(item['question_img_feature'])
            self.package_sequence.append(item['package_seq'])
            '''i+=1
            if i > 10:
                break'''

        self.data_total_length = len(self.question_img_feature)
        
        print(F'total_len = {self.data_total_length}')
        #self.drawTrajectoryDis()

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.question_img_feature[index], self.package_target[index], self.package_sequence[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.data_total_length

    def get_lens(self, sents):
        return [len(sent) for sent in sents]

    def drawTrajectoryDis(self):
        BOS_IDX = self.args.package_size + 2
        EOS_IDX = self.args.package_size + 3
        output = []
        for entry in self.package_sequence:
            entry = np.stack(entry) - 1
            #print(entry)
            #entry = np.concatenate((np.array(BOS_IDX).reshape(1,), entry, np.array(EOS_IDX).reshape(1,)))
            output.extend(entry.tolist())
            output.append(BOS_IDX)
            output.append(EOS_IDX)
        plt.hist(output, bins=31)
        plt.show()


# Create a dataloading module as per the PyTorch Lightning Docs
class SearchDataModule(pl.LightningDataModule):
  def __init__(self, args):
    super().__init__()
    train_set = FixDataset(args, 'Train')
    val_set = FixDataset(args, 'Valid')
    test_set = FixDataset(args, 'Test')
    collate_fn = Collator(args.package_size)
    
    self.train_loader = DataLoader(dataset=train_set,
                                    batch_size=args.batch_size,
                                    num_workers=2,
                                    collate_fn=collate_fn,
                                    shuffle=True)
    self.val_loader = DataLoader(dataset=val_set,
                                    batch_size=1,
                                    num_workers=2,
                                    collate_fn=collate_fn,
                                    shuffle=False)
    self.test_loader = DataLoader(dataset=test_set,
                                    batch_size=1,
                                    num_workers=2,
                                    collate_fn=collate_fn,
                                    shuffle=False)

  def train_dataloader(self):
    # dataiter = iter(self.train_loader)
    # train_images, labels = dataiter.next()
    # return train_images
    return self.train_loader

  def val_dataloader(self):
    # dataiter = iter(self.val_loader)
    # valid_images, labels = dataiter.next()
    # return valid_images
    return self.val_loader

  def test_dataloader(self):
    # dataiter = iter(self.test_loader)
    # test_images, labels = dataiter.next()
    # return test_images
    return self.test_loader

class Collator(object):
    def __init__(self, package_size):
        self.TGT_IDX = package_size
        self.PAD_IDX = package_size + 1
        self.BOS_IDX = package_size + 2
        self.EOS_IDX = package_size + 3

    def __call__(self, data):
        package_target = []
        package_seq = []
        question_img = []

        src_img = []
        tgt_img = []

        for data_entry in data:
            question_img_feature = data_entry[0]  # 27,300,186,3
            target = data_entry[1][0] - 1  # int
            gaze_seq = data_entry[2]  # 9,int
            gaze_seq = np.stack([gaze_seq]) - 1  # tgt, from 0-26
            gaze_seq = torch.from_numpy(gaze_seq).squeeze(0)

            gaze_seq = torch.cat((torch.tensor([self.BOS_IDX]),
                                  gaze_seq,
                                  torch.tensor([self.EOS_IDX])))
            package_seq.append(gaze_seq)
            target = torch.cat((torch.tensor([target]), torch.arange(27)))
            # target = torch.cat((torch.tensor([TGT_IDX]), torch.arange(27))) #CHANGE: Add TGT INDX
            package_target.append(target)
            question_img_feature = np.stack(question_img_feature)
            question_img_feature = torch.from_numpy(question_img_feature)
            # CHANGED to ones
            blank = torch.ones((4, question_img_feature.size()[1], question_img_feature.size()[2], 3))
            question_img.append(torch.cat((question_img_feature, blank), dim=0))  # 5,300,186,3

        package_seq = pad_sequence(package_seq, padding_value=self.PAD_IDX, batch_first=False)
        package_target = torch.stack(package_target).T
        question_img = torch.stack(question_img)
        # size: (b,31,w,h,3), (28, b), (max_len, b)
        # output: src_pos (28, b), src_img(b, 28, w, h, 3), tgt_pos(max_len, b), tgt_img(b, max_len, w, h, 3)
        batch_size = question_img.size()[0]
        for i in range(batch_size):
            indexes_src = package_target[:, i]
            imgs = question_img[i]  # 31, w, h, 3
            src_img_ = imgs[indexes_src]
            src_img.append(src_img_)
            tgt_img_ = imgs[package_seq[:, i]]
            tgt_img.append(tgt_img_)
        tgt_img = torch.stack(tgt_img)
        src_img = torch.stack(src_img)
        return package_target, src_img, package_seq, tgt_img
        # return question_img, package_target, package_seq




