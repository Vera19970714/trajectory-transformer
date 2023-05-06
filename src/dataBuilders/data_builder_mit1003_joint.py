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

        #assert len(self.imageData) == int(indexTxtList[0])
        foldImage = list(self.imageData)[int(indexTxtList[self.fold]):int(indexTxtList[self.fold+1])]

        #self.data_length = len(subjectData)
        #print(F'len = {self.data_length}')
        self.subject = []
        self.scanpath = []
        self.imageName = []

        self.imageRootPath = args.data_folder_path + 'ALLSTIMULI/'
        self.args = args

        i=0
        for item in subjectData:
            imageName = item['imagePath']
            if isTrain:  # exclude the subject
                if imageName not in foldImage:
                    self.subject.append(item['sub'])
                    self.scanpath.append(item['scanpathInPatch'])
                    self.imageName.append(item['imagePath'])
                    i += 1
            else:
                if imageName in foldImage:
                    self.subject.append(item['sub'])
                    self.scanpath.append(item['scanpathInPatch'])
                    self.imageName.append(item['imagePath'])
                    i += 1

            if i > 10:
               break

        self.data_total_length = len(self.subject)
        print(F'total_len = {self.data_total_length}')

    def __getitem__(self, index):
        return self.subject[index], self.scanpath[index], self.imageName[index], self.subject[index]

    def __len__(self):
        return self.data_total_length

    def get_lens(self, sents):
        return [len(sent) for sent in sents]

    def getImageData(self):
        return self.imageData


class MIT1003DataModule_JOINT(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        train_set = MIT1003Dataset(args, True)
        val_set = MIT1003Dataset(args, False)
        test_set = MIT1003Dataset(args, False)
        collate_fn_train = Collator(train_set.getImageData(), True, args.grid_partition, args.number_of_patches)
        collate_fn_test = Collator(train_set.getImageData(), False, args.grid_partition, args.number_of_patches)

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
    def __init__(self, imageData, isTrain, partitionGrid, number_of_patches):
        super().__init__()
        number_of_grids = int(partitionGrid * partitionGrid)
        self.package_size = int(number_of_patches * number_of_patches)
        self.PAD_IDX = number_of_grids
        self.BOS_IDX = self.PAD_IDX + 1
        self.EOS_IDX = self.PAD_IDX + 2
        self.total_extra_index = 3
        self.imageData = imageData
        self.heatmaps = imageData['heatmaps']
        self.isTrain = isTrain

    def __call__(self, data):

        package_target = []
        package_seq = []
        question_img = []

        src_img = []
        tgt_img = []
        heatmaps = []
        firstImageName = data[0][2]

        for data_entry in data:
            imageName = data_entry[2]
            if not self.isTrain:
                assert firstImageName == imageName
            question_img_feature = self.imageData[imageName]
            heatmap = self.heatmaps[imageName]
            gaze_seq = data_entry[1]
            gaze_seq = torch.from_numpy(gaze_seq).squeeze(0)

            gaze_seq = torch.cat((torch.tensor([self.BOS_IDX]),
                                  gaze_seq, torch.tensor([self.EOS_IDX])))
            package_seq.append(gaze_seq)
            target = torch.arange(self.package_size)
            package_target.append(target)

            heatmap = torch.from_numpy(heatmap).squeeze(0)
            heatmaps.append(heatmap)

            question_img_feature = np.stack(question_img_feature)
            question_img_feature = torch.from_numpy(question_img_feature)
            blank = torch.ones((self.total_extra_index, question_img_feature.size()[1], question_img_feature.size()[2],
                                question_img_feature.size()[3]))
            question_img.append(torch.cat((question_img_feature, blank), dim=0).float())  # 5,300,186,3

        package_seq = pad_sequence(package_seq, padding_value=self.PAD_IDX, batch_first=False)
        package_target = torch.stack(package_target).T

        batch_size = len(question_img)
        for i in range(batch_size):
            indexes_src = package_target[:, i]
            imgs = question_img[i]  # 31, w, h, 3
            src_img_ = imgs[indexes_src]
            src_img.append(src_img_)
            tgt_img_ = imgs[package_seq[:, i]]
            tgt_img.append(tgt_img_)
        if self.isTrain:
            return package_target, src_img, package_seq, tgt_img, heatmaps
        else:  # extra image name for SPP evaluation
            return firstImageName, package_target, src_img, package_seq, tgt_img, heatmaps
