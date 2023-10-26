import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pytorch_lightning as pl
import pickle
from torch.nn.utils.rnn import pad_sequence
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class IrregularDataset(Dataset):
    def __init__(self, datapath):
        print('Loading irregular shelf')
        with open(datapath, "rb") as fp:
            raw_data = pickle.load(fp)

        self.data_length = len(raw_data)
        print(F'initial len = {self.data_length}')
        self.package_target = []
        self.question_img_feature = []
        self.package_sequence = []
        self.ids = []

        # i=0
        for item in raw_data:
            if item == {}:
                continue
            self.package_target.append(item['package_target'])
            self.question_img_feature.append(item['question_img_feature'])
            self.package_sequence.append(item['package_seq'])
            #self.ids.append(item['id'])
            '''i+=1
            if i > 10:
                break'''

        self.data_total_length = len(self.question_img_feature)

        print(F'final len = {self.data_total_length}')
        # self.drawTrajectoryDis()

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.question_img_feature[index], self.package_target[index], self.package_sequence[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.data_total_length

    def get_lens(self, sents):
        return [len(sent) for sent in sents]

# Create a dataloading module as per the PyTorch Lightning Docs
class IrregularShelfModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        dataset = IrregularDataset('./dataset/processdata/dataset_irregular_yes')
        collate_fn = Collator_pure(43)

        self.test_loader = DataLoader(dataset=dataset,
                                      batch_size=1,
                                      num_workers=2,
                                      collate_fn=collate_fn,
                                      shuffle=False)


class Collator_pure(object):
    def __init__(self, package_size):
        # self.TGT_IDX = package_size
        self.PAD_IDX = package_size + 1  # 1
        self.BOS_IDX = package_size + 2
        self.EOS_IDX = package_size  # + 3
        self.package_size = package_size
        # EOS index: 2, 5; PAD index: -1, -1; BOS index: 2, 5

    def __call__(self, data):
        package_target = []
        package_seq = []

        src_img = []
        tgt_img = []

        batch_size = len(data)
        assert batch_size == 1
        data_entry = data[0]

        question_img_feature = data_entry[0]  # 27,300,186,3
        target = data_entry[1][0] - 1  # int
        gaze_seq = data_entry[2]  # 9,int
        gaze_seq = np.stack([gaze_seq]) - 1  # tgt, from 0-26
        gaze_seq = torch.from_numpy(gaze_seq).squeeze(0)

        gaze_seq = torch.cat((torch.tensor([self.BOS_IDX]),
                              gaze_seq,
                              torch.tensor([self.EOS_IDX])))
        package_seq.append(gaze_seq)
        target = torch.cat((torch.arange(self.package_size), torch.tensor([target])))
        package_target.append(target)
        question_img = question_img_feature + [(np.ones((128, 50, 3)), np.array([2, 5]))] + \
                       [(np.ones((128, 50, 3)), np.array([-1, -1]))] + [(np.ones((128, 50, 3)), np.array([2, 5]))]

        package_seq = pad_sequence(package_seq, padding_value=self.PAD_IDX, batch_first=False)
        package_target = torch.stack(package_target).T

        i = 0
        img_ = []
        ind_ = []
        for j in range(self.package_size+1):
            pac = package_target[j][0]
            img_.append(torch.from_numpy(question_img[pac][0]).unsqueeze(0).float().to(DEVICE))
            ind_.append(torch.from_numpy(question_img[pac][1]-1).to(DEVICE))
        src_img.append(img_)
        src_pos = torch.stack(ind_).unsqueeze(-1)

        img_ = []
        ind_ = []
        for j in range(package_seq.size()[0]):
            pac = package_seq[j][0]
            img_.append(torch.from_numpy(question_img[pac][0]).unsqueeze(0).float().to(DEVICE))
            ind_.append(torch.from_numpy(question_img[pac][1]-1).to(DEVICE))
        tgt_img.append(img_)
        tgt_pos = torch.stack(ind_).unsqueeze(-1)

        return src_pos, src_img, tgt_pos, tgt_img, package_seq
        # output: src_pos (44, 2, b), src_img(b, 44, w, h, 3), tgt_pos(max_len, 2, b), tgt_img(b, max_len, w, h, 3)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-data_path', default='./dataset/processdata/dataset_irregular_yes', type=str)
    parser.add_argument('-package_size', default=43, type=int)
    args = parser.parse_args()

    #data = IrregularDataset(args)
    dataloader = IrregularShelfModule(args)
    for x in enumerate(dataloader.test_loader):
        break
