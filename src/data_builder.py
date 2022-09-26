import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
import pytorch_lightning as pl

class FixDataset(Dataset):
    def __init__(self, args, new_datapath):
        raw_data = torch.load(new_datapath)
        self.data_length = len(raw_data)
        print(F'len = {self.data_length}')
        self.package_target = []
        self.question_img_feature = []
        self.package_sequence = []
        self.args = args

        for item in raw_data:
            self.package_target.append(item['package_target'])
            self.question_img_feature.append(item['question_img_feature'])
            self.package_sequence.append(item['package_seq'])
            
        self.data_total_length = len(self.question_img_feature)
        
        print(F'total_len = {self.data_total_length}')
        
       

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.question_img_feature[index], self.package_target[index], self.package_sequence[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.data_total_length


    def get_lens(self, sents):
        return [len(sent) for sent in sents]


# Create a dataloading module as per the PyTorch Lightning Docs
class SearchDataModule(pl.LightningDataModule):
  def __init__(self, args):
    super().__init__()
    train_set = FixDataset(args, args.train_datapath)
    val_set = FixDataset(args, args.valid_datapath)
    test_set = FixDataset(args, args.test_datapath)
    
    self.train_loader = DataLoader(dataset=train_set, \
                                    batch_size=args.batch_size, \
                                    num_workers=2, \
                                    shuffle=True)
    self.val_loader = DataLoader(dataset=val_set, \
                                    batch_size=args.batch_size, \
                                    num_workers=2, \
                                    shuffle=False)
    self.test_loader = DataLoader(dataset=test_set, \
                                    batch_size=args.batch_size, \
                                    num_workers=2, \
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