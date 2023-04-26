import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pandas as pd
import pickle
from torch.nn.utils.rnn import pad_sequence
import numpy as np

class MIT1003Dataset(Dataset):
    def __init__(self, args, isTrain):
        data_folder_path = args.data_folder_path
        dataPath = data_folder_path + args.processed_data_name
        self.fold = args.fold
        #allSubjects = ['CNG', 'ajs', 'emb', 'ems', 'ff', 'hp', 'jcw', 'jw', 'kae', 'krl', 'po', 'tmj', 'tu', 'ya', 'zb']
        with open(dataPath, "rb") as fp:  # Unpickling
            raw_data = pickle.load(fp)

        subjectData = raw_data[:-1]
        self.imageData = raw_data[-1]

        indexTxtFilePath = data_folder_path + 'crossValidationIndex.txt'
        indexTxtFile = open(indexTxtFilePath, "r")
        indexTxt = indexTxtFile.read()
        indexTxtList = indexTxt.split("\n")
        indexTxtFile.close()

        assert len(self.imageData) == int(indexTxtList[0])
        foldImage = list(self.imageData)[int(indexTxtList[self.fold]):int(indexTxtList[self.fold+1])]

        #self.data_length = len(subjectData)
        #print(F'len = {self.data_length}')
        self.subject = []
        self.scanpath = []
        self.heatmap = []
        self.imageName = []

        self.imageRootPath = args.data_folder_path + 'ALLSTIMULI/'
        self.args = args

        # i=0
        for item in subjectData:
            imageName = item['imagePath']
            if isTrain:  # exclude the subject
                if imageName not in foldImage:
                    self.subject.append(item['sub'])
                    self.scanpath.append(item['scanpathInPatch'])
                    self.imageName.append(item['imagePath'])
                    self.heatmap.append(item['heatmap'])
                    # i += 1
            else:
                if imageName in foldImage:
                    self.subject.append(item['sub'])
                    self.scanpath.append(item['scanpathInPatch'])
                    self.imageName.append(item['imagePath'])
                    self.heatmap.append(item['heatmap'])
                    # i += 1

            # if i > 32:
            #    break

        self.data_total_length = len(self.subject)
        print(F'total_len = {self.data_total_length}')

    def __getitem__(self, index):
        return self.subject[index], self.scanpath[index], self.imageName[index], self.heatmap[index]

    def __len__(self):
        return self.data_total_length

    def get_lens(self, sents):
        return [len(sent) for sent in sents]

    def getImageData(self):
        return self.imageData


class MIT1003DataModule_VIT(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        train_set = MIT1003Dataset(args, True)
        val_set = MIT1003Dataset(args, False)
        test_set = MIT1003Dataset(args, False)
        collate_fn_train = Collator(train_set.getImageData(), True, args.grid_partition)
        collate_fn_test = Collator(train_set.getImageData(), False, args.grid_partition)

        self.train_loader = DataLoader(dataset=train_set,
                                       batch_size=args.batch_size,
                                       num_workers=2,
                                       collate_fn=collate_fn_train,
                                       shuffle=True)
        self.val_loader = DataLoader(dataset=val_set,
                                     batch_size=args.batch_size,
                                     num_workers=2,
                                     collate_fn=collate_fn_train,
                                     shuffle=False)
        self.test_loader = DataLoader(dataset=test_set,
                                      batch_size=1, #SHOULD ALWAYS BE 1
                                      num_workers=2,
                                      collate_fn=collate_fn_test,
                                      shuffle=False)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader


class Collator(object):
    def __init__(self, imageData, isTrain, partitionGrid):
        super().__init__()
        if partitionGrid != -1:
            self.package_size = int(partitionGrid * partitionGrid)
        else:
            self.package_size = 28
        self.PAD_IDX = self.package_size
        self.BOS_IDX = self.PAD_IDX+1
        self.EOS_IDX = self.PAD_IDX+2
        self.total_extra_index = 3
        self.imageData = imageData
        self.isTrain = isTrain

    def __call__(self, data):
        package_seq = []
        scanpath_seq = []
        src_img = []
        heatmaps = []
        firstImageName = data[0][2]

        for data_entry in data:
            imageName = data_entry[2]
            image = torch.from_numpy(self.imageData[imageName])
            src_img.append(image)

            if not self.isTrain:
                assert firstImageName == imageName
            gaze_seq = data_entry[1]
            gaze_seq = torch.from_numpy(gaze_seq).squeeze(0)
            gaze_seq = torch.cat((torch.tensor([self.BOS_IDX]), gaze_seq)) #, torch.tensor([self.EOS_IDX])))
            package_seq.append(gaze_seq)
            heatmap = data_entry[3]
            heatmap = torch.from_numpy(heatmap).squeeze(0)
            heatmaps.append(heatmap)

        package_seq = pad_sequence(package_seq, padding_value=self.PAD_IDX, batch_first=False)

        # tgt_img and src_img are lists due to variant size
        src_img = torch.stack(src_img)

        # output: src_img(b, w, h, 3), tgt_pos(max_len, b)
        return firstImageName, src_img, package_seq, heatmaps




