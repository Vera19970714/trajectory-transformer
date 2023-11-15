import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pytorch_lightning as pl
import pickle
from torch.nn.utils.rnn import pad_sequence


def get_val_and_tst(testing_dataset):
    split_num_valid = int(len(testing_dataset) * 0.8)
    split_num_test = int(len(testing_dataset) * 0.9)
    val = testing_dataset[split_num_valid:split_num_test]
    tst = testing_dataset[split_num_test:]
    return val, tst

def randsplit(datapath, indexFile, isTrain, testing_dataset_choice, training_dataset_choice):
    with open(datapath, "rb") as fp:
        raw_data = pickle.load(fp)

    with open(indexFile) as f:
        lines = f.readlines()
    linesInt = np.array([int(x) for x in lines])

    raw_data = np.array(raw_data)[linesInt.astype(int)]

    shampoo_task = []
    yogurt_task = []
    wine_task = []
    for index in range(len(raw_data)):
        if raw_data[index]['id'] == 'Q2':
            shampoo_task.append(index)
        elif raw_data[index]['id'] == 'Q3':
            yogurt_task.append(index)
        elif raw_data[index]['id'] == 'Q1':
            wine_task.append(index)

    val_wine, tst_wine = get_val_and_tst(wine_task)
    val_yogurt, tst_yogurt = get_val_and_tst(yogurt_task)

    if testing_dataset_choice == 'wine':
        val = val_wine
        tst = tst_wine
    elif testing_dataset_choice == 'yogurt':
        val = val_yogurt
        tst = tst_yogurt
    else:
        val = val_wine + val_yogurt
        tst = tst_wine + tst_yogurt

    if training_dataset_choice == 'wine':
        training_dataset = wine_task
    elif training_dataset_choice == 'yogurt':
        training_dataset = yogurt_task
    else:
        training_dataset = wine_task + yogurt_task

    if isTrain == 'Valid':
        return raw_data[np.array(val).astype(int)]
    if isTrain == 'Test':
        return raw_data[np.array(tst).astype(int)]

    final_training = []
    for i in training_dataset:
        if i not in val and i not in tst:
            final_training.append(i)
    if isTrain == 'Train':
        return raw_data[np.array(final_training).astype(int)]


class FixDataset(Dataset):
    def __init__(self, args, isTrain):
        datapath = args.data_path
        indexFile = args.index_folder + args.index_file
        testing_dataset_choice = args.testing_dataset_choice
        training_dataset_choice = args.training_dataset_choice

        assert testing_dataset_choice in ["wine", "yogurt", "all", "irregular"]
        assert training_dataset_choice in ["wine", "yogurt", "all"]
        print('Settings: ', isTrain, training_dataset_choice, testing_dataset_choice)


        raw_data = randsplit(datapath, indexFile, isTrain, testing_dataset_choice, training_dataset_choice)


        self.data_length = len(raw_data)
        print(F'len = {self.data_length}')
        self.package_target = []
        self.question_img_feature = []
        self.package_sequence = []
        self.args = args
        self.ids = []
        self.targets = []
        self.layouts = []
        self.max_len = 0

        #i=0
        #print('change it back')
        for item in raw_data:
            self.package_target.append(item['package_target'])
            self.question_img_feature.append(item['question_img_feature'])
            self.package_sequence.append(item['package_seq'])
            self.ids.append(item['id'])
            self.targets.append(item['tgt_id'])
            self.layouts.append(item['layout_id'])
            if len(item['package_seq']) > self.max_len:
                self.max_len = len(item['package_seq'])
            '''i+=1
            if i > 3:
                break'''

        self.data_total_length = len(self.question_img_feature)
        print(F'total_len = {self.data_total_length}, ', 'max len=', self.max_len)
        # self.drawTrajectoryDis()

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.question_img_feature[index], self.package_target[index], self.package_sequence[index], self.ids[
            index], self.targets[index], self.layouts[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.data_total_length

    def get_lens(self, sents):
        return [len(sent) for sent in sents]



# Create a dataloading module as per the PyTorch Lightning Docs
class GazeformerDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        train_set = FixDataset(args, 'Train')
        self.max_len = train_set.max_len
        val_set = FixDataset(args, 'Valid')
        test_set = FixDataset(args, 'Test')
        collate_fn = Collator_mixed(args.package_size)

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
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader


class Collator_mixed(object):
    def __init__(self, package_size):
        # self.TGT_IDX = package_size
        self.PAD_IDX = package_size + 1  # 1
        self.BOS_IDX = package_size + 2
        self.EOS_IDX = package_size  # + 3
        self.package_size = package_size

    def process_one_type(self, data, type_index):  # type index: 0 is yogurt, 1 is wine
        package_target = []
        package_seq = []
        #question_img = []

        src_img = []
        tgt_img = []

        for data_entry in data:
            target_feature = np.load('./dataset/img/Target_features/'+data_entry[4]+'.npy') # 1, 15, 2048
            layout_feature = np.load('./dataset/img/Question_features/'+data_entry[5]+'.npy') # 1, 640, 2048
            src_img.append(layout_feature)
            tgt_img.append(target_feature)
            #question_img_feature = data_entry[0]  # 27,300,186,3
            target = data_entry[1][0] - 1  # int
            gaze_seq = data_entry[2]  # 9,int
            gaze_seq = np.stack([gaze_seq]) - 1  # tgt, from 0-26
            gaze_seq = torch.from_numpy(gaze_seq).squeeze(0)

            gaze_seq = torch.cat((torch.tensor([self.BOS_IDX[type_index]]),
                                  gaze_seq,
                                  torch.tensor([self.EOS_IDX[type_index]])))
            package_seq.append(gaze_seq)
            target = torch.cat((torch.arange(self.package_size[type_index]), torch.tensor([target])))
            package_target.append(target)

        package_seq = pad_sequence(package_seq, padding_value=self.PAD_IDX[type_index], batch_first=False)
        package_target = torch.stack(package_target).T
        tgt_img = torch.from_numpy(np.stack(tgt_img))
        src_img = torch.from_numpy(np.stack(src_img))
        # databuilder output: src 10, 640, 2048; firstfix: 10, 2; task: 10, 768; token_prob: 7,10,2;
        return package_target, src_img, package_seq, tgt_img
        # package_target: 23, 1; src_img: 1, 640, 2048; package_seq: 9, 1; tgt_img: 1, 15, 2048; token_prob: 9, 1

    def __call__(self, data):
        yogurt_data = []
        wine_data = []
        for data_entry in data:
            id = data_entry[3]
            if id == 'Q1':
                wine_data.append(data_entry)
            elif id == 'Q3':
                yogurt_data.append(data_entry)
        if len(yogurt_data) == 0:
            data1 = []
        else:
            data1 = self.process_one_type(yogurt_data, 0)
        if len(wine_data) == 0:
            data2 = []
        else:
            data2 = self.process_one_type(wine_data, 1)
        return data1, data2




