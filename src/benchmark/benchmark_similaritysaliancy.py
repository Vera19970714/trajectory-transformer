import numpy as np
from torch.utils.data import DataLoader
from data_builder import SimilarityFixDataset, collate_fn_similarity
import torch
from scipy.stats import rv_discrete
import pandas as pd
from random import randint
# seed random number generator
#seed(1)

# TODO: carefully fit the number accordingly
avg_len = 7.7
minLen = 1
iter = 100
test_datapath = '../dataset/processdata/dataset_Q23_similarity_mousedel_time_val'
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 27, 28, 29, 30
TOTAL_PCK = 27

# ------------------------------------------------------
end_pro = 1/avg_len
end_prob = 1 / (avg_len+1) * 100
DEVICE = torch.device('cpu')
test_set = SimilarityFixDataset(0, test_datapath)
test_loader = DataLoader(dataset=test_set, batch_size=1, num_workers=0, collate_fn=collate_fn_similarity, shuffle=False)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
losses1 = 0
losses2 = 0
losses3 = 0
#max_length = 17
time = 0

softmax = torch.nn.Softmax(dim=0)
all_gaze_similarity, all_gaze_saliency, all_gaze_rgb = pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
for package_similarity, package_saliency, package_rgb, package_seq in test_loader:
    time += 1
    package_similarity = package_similarity.to(DEVICE)
    package_saliency = package_saliency.to(DEVICE)
    package_rgb = package_rgb.to(DEVICE)
    package_seq = package_seq.to(DEVICE)
    
    tgt_out = package_seq[1:, :]
    length = tgt_out.size(0)

    similarity_dis = torch.cat((package_similarity,torch.tensor([0, 0, 0, end_pro]).view(1, -1).to(DEVICE)),1)
    saliency_dis = torch.cat((package_saliency,torch.tensor([0, 0, 0, end_pro]).view(1, -1).to(DEVICE)),1)
    rgb_dis = torch.cat((package_rgb,torch.tensor([0, 0, 0, end_pro]).view(1, -1).to(DEVICE)),1)
    output_similarity = similarity_dis.repeat(tgt_out.size()[0], 1)
    output_saliency = saliency_dis.repeat(tgt_out.size()[0], 1)
    output_rgb = rgb_dis.repeat(tgt_out.size()[0], 1)

    sim = softmax(similarity_dis[0, :TOTAL_PCK])
    sal = softmax(saliency_dis[0, :TOTAL_PCK])
    rgb = softmax(rgb_dis[0, :TOTAL_PCK])

    for n in range(iter):
        GAZE_similarity = []
        GAZE_saliency = []
        GAZE_rgb = []

        x = randint(0, 101)
        while x >= end_prob or len(GAZE_similarity)<minLen:
            ind = np.random.choice(TOTAL_PCK,1,p=sim.numpy())
            GAZE_similarity.append(ind)
            x = randint(0, 101)
        gaze_df = np.stack(GAZE_similarity).reshape(1, -1)
        all_gaze_similarity = pd.concat([all_gaze_similarity, pd.DataFrame(gaze_df)],axis=0)

        x = randint(0, 101)
        while x >= end_prob or len(GAZE_saliency) < minLen:
            ind = np.random.choice(TOTAL_PCK, 1, p=sal.numpy())
            GAZE_saliency.append(ind)
            x = randint(0, 101)
        gaze_df = np.stack(GAZE_saliency).reshape(1, -1)
        all_gaze_saliency = pd.concat([all_gaze_saliency, pd.DataFrame(gaze_df)], axis=0)

        x = randint(0, 101)
        while x >= end_prob or len(GAZE_rgb) < minLen:
            ind = np.random.choice(TOTAL_PCK, 1, p=rgb.numpy())
            GAZE_rgb.append(ind)
            x = randint(0, 101)
        gaze_df = np.stack(GAZE_rgb).reshape(1, -1)
        all_gaze_rgb = pd.concat([all_gaze_rgb, pd.DataFrame(gaze_df)], axis=0)
    
    loss1 = loss_fn(output_similarity, tgt_out.reshape(-1))
    loss2 = loss_fn(output_saliency, tgt_out.reshape(-1))
    loss3 = loss_fn(output_rgb, tgt_out.reshape(-1))

    losses1 += loss1.item()
    losses2 += loss2.item()
    losses3 += loss3.item()
print('similarity loss:',losses1 / time)
print('saliency loss:',losses2 / time)
print('rgb loss:',losses3 / time)

all_gaze_similarity.to_csv('../dataset/checkEvaluation/gaze_resnet_similarity.csv', index=False)
all_gaze_saliency.to_csv('../dataset/checkEvaluation/gaze_saliency.csv', index=False)
all_gaze_rgb.to_csv('../dataset/checkEvaluation/gaze_rgb_similarity.csv', index=False)