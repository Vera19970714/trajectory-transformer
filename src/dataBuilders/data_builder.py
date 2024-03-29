import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
import pytorch_lightning as pl
import pickle
from torch.nn.utils.rnn import pad_sequence
#import matplotlib.pyplot as plt


class FixDataset(Dataset):
    def __init__(self, args, new_datapath):
        #raw_data = torch.load(new_datapath)
        with open(new_datapath, "rb") as fp:  # Unpickling
            raw_data = pickle.load(fp)
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
    args.package_size = 27
    train_set = FixDataset(args, args.train_datapath)
    val_set = FixDataset(args, args.valid_datapath)
    test_set = FixDataset(args, args.test_datapath)
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




