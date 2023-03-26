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

        # calculate how many kinds of image sizes
        '''allSizes = {}
        for element in list(self.imageData):
            feature = self.imageData[element]
            name = str(feature.shape[1])+','+str(feature.shape[2])
            allSizes[name] = 0
        print(allSizes)
        print(len(allSizes))
        quit()'''

        indexTxtFilePath = data_folder_path + 'crossValidationIndex.txt'
        indexTxtFile = open(indexTxtFilePath, "r")
        indexTxt = indexTxtFile.read()
        indexTxtList = indexTxt.split("\n")
        indexTxtFile.close()

        assert len(self.imageData) == int(indexTxtList[0])
        foldImage = list(self.imageData)[int(indexTxtList[self.fold]):int(indexTxtList[self.fold+1])]

        # not used
        '''indexs = np.unravel_index(np.arange(N * N), (N, N))  # size: 16, 2
        indexs = np.concatenate((indexs[0].reshape(1, -1), indexs[1].reshape(1, -1)), axis=0)
        self.indices = indexs'''

        #self.data_length = len(subjectData)
        #print(F'len = {self.data_length}')
        self.subject = []
        self.scanpath = []
        self.scanpathPixel = []
        self.imageSize = []
        self.imageName = []
        self.patchIndex = []
        self.args = args

        i=0
        for item in subjectData:
            imageName = item['imagePath']
            if isTrain:  # exclude the subject
                #if item['sub'] != subject:
                if imageName not in foldImage:
                    self.subject.append(item['sub'])
                    self.scanpath.append(item['scanpathInPatch'])
                    self.scanpathPixel.append(item['scanpath'])
                    if self.args.saliency_metric == 'True':
                        self.imageSize.append(item['imageSize'])
                    self.imageName.append(item['imagePath'])
                    #self.patchIndex.append(self.indices)
                    i += 1
            else:
                #if item['sub'] == subject:  # only include the subject
                if imageName in foldImage:
                    self.subject.append(item['sub'])
                    self.scanpath.append(item['scanpathInPatch'])
                    self.scanpathPixel.append(item['scanpath'])
                    if self.args.saliency_metric == 'True':
                        self.imageSize.append(item['imageSize'])
                    self.imageName.append(item['imagePath'])
                    #self.patchIndex.append(self.indices)
                    i += 1

            #if i > 10:
            #   break

        self.data_total_length = len(self.subject)

        print(F'total_len = {self.data_total_length}')

    def __getitem__(self, index):
        if self.args.saliency_metric == 'True':
            return self.subject[index], self.scanpath[index], self.imageName[index], self.subject[index],self.scanpathPixel[index],self.imageSize[index] #, self.patchIndex[index]
        else:
            return self.subject[index], self.scanpath[index], self.imageName[index], self.subject[index], \
                   self.scanpathPixel[index]

    def __len__(self):
        return self.data_total_length

    def get_lens(self, sents):
        return [len(sent) for sent in sents]

    def getImageData(self):
        return self.imageData


class MIT1003DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        train_set = MIT1003Dataset(args, True)
        val_set = MIT1003Dataset(args, False)
        test_set = MIT1003Dataset(args, False)
        collate_fn_train = Collator(train_set.getImageData(), True, args.grid_partition, args)
        collate_fn_test = Collator(train_set.getImageData(), False, args.grid_partition, args)

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
    def __init__(self, imageData, isTrain, partitionGrid, args):
        super().__init__()
        self.args = args
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
        package_target = []
        package_seq = []
        scanpath_seq = []
        question_img = []

        src_img = []
        tgt_img = []

        firstImageName = data[0][2]
        if self.args.saliency_metric == 'True':
            firstImgSize = data[0][5]

        for data_entry in data:
            if self.args.saliency_metric == 'True':
                imgSize = data_entry[5]
            imageName = data_entry[2]
            scanpath = data_entry[4]
            if not self.isTrain:
                assert firstImageName == imageName
                if self.args.saliency_metric == 'True':
                    assert firstImgSize == imgSize
            question_img_feature = self.imageData[imageName]
            gaze_seq = data_entry[1]
            gaze_seq = torch.from_numpy(gaze_seq).squeeze(0)

            gaze_seq = torch.cat((torch.tensor([self.BOS_IDX]),
                              gaze_seq,
                              torch.tensor([self.EOS_IDX])))
            scanpath = torch.from_numpy(np.array(scanpath)).squeeze(0)
            package_seq.append(gaze_seq)
            scanpath_seq.append(scanpath)
            target = torch.arange(self.package_size)
            package_target.append(target)
            question_img_feature = np.stack(question_img_feature)
            question_img_feature = torch.from_numpy(question_img_feature)
            # CHANGED to ones
            blank = torch.ones((self.total_extra_index, question_img_feature.size()[1], question_img_feature.size()[2], 3))
            question_img.append(torch.cat((question_img_feature, blank), dim=0))  # 5,300,186,3

        package_seq = pad_sequence(package_seq, padding_value=self.PAD_IDX, batch_first=False)
        package_target = torch.stack(package_target).T
        scanpath_seq = pad_sequence(scanpath_seq, padding_value=self.PAD_IDX, batch_first=False)
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
        if self.args.saliency_metric == 'True':
            firstImgSize = torch.from_numpy(np.array(firstImgSize))

        if self.isTrain:
            return package_target, src_img, package_seq, tgt_img
        else:  # extra image name for SPP evaluation
            if self.args.saliency_metric == 'True':
                return firstImageName, firstImgSize, package_target, src_img, package_seq, tgt_img, scanpath_seq
            else:
                return firstImageName, package_target, src_img, package_seq, tgt_img, scanpath_seq


if __name__ == '__main__':
    class ARGS(object):
        def __init__(self):
            self.fold = 1
            self.data_folder_path = '../dataset/MIT1003/'
            self.processed_data_name = 'processedData_N4_centerMode'
            self.grid_partition = -1
            self.batch_size = 10
    args = ARGS()
    '''mit = MIT1003Dataset(args, False)
    collate_fn = Collator(mit.getImageData())
    train_loader = DataLoader(dataset=mit,
                              batch_size=1,
                              num_workers=0,
                              collate_fn=collate_fn,
                              shuffle=True)'''
    mit = MIT1003DataModule(args)
    train_loader = mit.train_loader
    for batch in train_loader:
        print(batch[2].flatten())


