import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pandas as pd
import pickle
from torch.nn.utils.rnn import pad_sequence
import numpy as np

class MIT1003Dataset(Dataset):
    def __init__(self, args, subject, isTrain, dataPath='../dataset/MIT1003/processedData'):
        #allSubjects = ['CNG', 'ajs', 'emb', 'ems', 'ff', 'hp', 'jcw', 'jw', 'kae', 'krl', 'po', 'tmj', 'tu', 'ya', 'zb']
        with open(dataPath, "rb") as fp:  # Unpickling
            raw_data = pickle.load(fp)
        self.data_length = len(raw_data)
        print(F'len = {self.data_length}')
        self.subject = []
        self.scanpath = []
        self.imageFeature = []
        self.patchIndex = []
        self.args = args

        # i=0
        for item in raw_data:
            if isTrain:  # exclude the subject
                if item['sub'] != subject:
                    self.subject.append(item['sub'])
                    self.scanpath.append(item['scanpathInPatch'])
                    self.imageFeature.append(item['imageFeature'])
                    self.patchIndex.append(item['patchIndex'])
            else:
                if item['sub'] == subject:  # only include the subject
                    self.subject.append(item['sub'])
                    self.scanpath.append(item['scanpathInPatch'])
                    self.imageFeature.append(item['imageFeature'])
                    self.patchIndex.append(item['patchIndex'])
            '''i+=1
            if i > 10:
                break'''

        self.data_total_length = len(self.subject)

        print(F'total_len = {self.data_total_length}')

    def __getitem__(self, index):
        return self.subject[index], self.scanpath[index], self.imageFeature[index], self.patchIndex[index]

    def __len__(self):
        return self.data_total_length

    def get_lens(self, sents):
        return [len(sent) for sent in sents]


class MIT1003DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        train_set = MIT1003Dataset(args, args.subject, True, args.datapath)
        val_set = MIT1003Dataset(args, args.subject, False, args.datapath)
        test_set = MIT1003Dataset(args, args.subject, False, args.datapath)
        collate_fn = Collator()

        self.train_loader = DataLoader(dataset=train_set,
                                       batch_size=args.batch_size,
                                       num_workers=2,
                                       collate_fn=collate_fn,
                                       shuffle=True)
        self.val_loader = DataLoader(dataset=val_set,
                                     batch_size=1, #SHOULD ALWAYS BE 1
                                     num_workers=2,
                                     collate_fn=collate_fn,
                                     shuffle=False)
        self.test_loader = DataLoader(dataset=test_set,
                                      batch_size=1, #SHOULD ALWAYS BE 1
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
    def __init__(self):
        super().__init__()
        self.PAD_IDX = 16
        self.BOS_IDX = 17
        self.EOS_IDX = 18
        self.total_extra_index = 3
        self.package_size = 16

    def __call__(self, data):
        package_target = []
        package_seq = []
        question_img = []

        src_img = []
        tgt_img = []

        for data_entry in data:
            question_img_feature = data_entry[2]  # 27,300,186,3
            gaze_seq = data_entry[1]
            gaze_seq = torch.from_numpy(gaze_seq).squeeze(0)

            gaze_seq = torch.cat((torch.tensor([self.BOS_IDX]),
                                  gaze_seq,
                                  torch.tensor([self.EOS_IDX])))
            package_seq.append(gaze_seq)
            target = torch.arange(self.package_size)
            package_target.append(target)
            question_img_feature = np.stack(question_img_feature)
            question_img_feature = torch.from_numpy(question_img_feature)
            # CHANGED to ones
            blank = torch.ones((self.total_extra_index, question_img_feature.size()[1], question_img_feature.size()[2], 3))
            question_img.append(torch.cat((question_img_feature, blank), dim=0))  # 5,300,186,3

        package_seq = pad_sequence(package_seq, padding_value=self.PAD_IDX, batch_first=False)
        package_target = torch.stack(package_target).T
        #question_img = torch.stack(question_img)

        batch_size = len(question_img)
        for i in range(batch_size):
            indexes_src = package_target[:, i]
            imgs = question_img[i]  # 31, w, h, 3
            src_img_ = imgs[indexes_src]
            src_img.append(src_img_)
            tgt_img_ = imgs[package_seq[:, i]]
            tgt_img.append(tgt_img_)
        # tgt_img and src_img are lists due to variant size
        #tgt_img = torch.stack(tgt_img)
        #src_img = torch.stack(src_img)
        # output: src_pos (16, b), src_img(b, 16, w, h, 3), tgt_pos(max_len, b), tgt_img(b, max_len, w, h, 3)
        return package_target, src_img, package_seq, tgt_img



if __name__ == '__main__':
    mit = MIT1003Dataset(0, 'emb', True)
    collate_fn = Collator()
    train_loader = DataLoader(dataset=mit,
                              batch_size=2,
                              num_workers=0,
                              collate_fn=collate_fn,
                              shuffle=True)
    for batch in train_loader:
        print(batch)


