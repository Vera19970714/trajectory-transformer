import numpy as np
from torch.utils.data import DataLoader
from data_builder import SimilarityFixDataset, collate_fn_similarity
import torch
from scipy.stats import rv_discrete
import pandas as pd

# seed random number generator
#seed(1)
avg_len = 7.71
end_pro = 0.13
continue_pro = 0.87

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 27, 28, 29, 30
DEVICE = torch.device('cpu')
test_datapath = './dataset/processdata/dataset_Q23_similarity_mousedel_time_val'
test_set = SimilarityFixDataset(0, test_datapath)
test_loader = DataLoader(dataset=test_set, batch_size=1, num_workers=0, collate_fn=collate_fn_similarity, shuffle=False)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
losses1 = 0
losses2 = 0
losses3 = 0
max_length = 17
time = 0
iter = 100
xk = np.arange(31)
all_gaze_similarity, all_gaze_saliency, all_gaze_rgb = pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
for package_similarity, package_saliency, package_rgb, package_seq in test_loader:
    time += 1
    package_similarity = package_similarity.to(DEVICE)
    package_saliency = package_saliency.to(DEVICE)
    package_rgb = package_rgb.to(DEVICE)
    package_seq = package_seq.to(DEVICE)
    
    tgt_out = package_seq[1:, :]
    length = tgt_out.size(0)
    GAZE_similarity = np.zeros((max_length, iter))-1
    GAZE_saliency = np.zeros((max_length, iter))-1
    GAZE_rgb = np.zeros((max_length, iter))-1
    similarity_dis = (continue_pro/torch.sum(package_similarity))*package_similarity
    saliency_dis = (continue_pro/torch.sum(package_saliency))*package_saliency
    rgb_dis = (continue_pro/torch.sum(package_rgb))*package_rgb
    # 1, 28
    similarity_dis = torch.cat((similarity_dis,torch.tensor([0, 0, 0, end_pro]).view(1, -1).to(DEVICE)),1)
    saliency_dis = torch.cat((saliency_dis,torch.tensor([0, 0, 0, end_pro]).view(1, -1).to(DEVICE)),1)
    rgb_dis = torch.cat((rgb_dis,torch.tensor([0, 0, 0, end_pro]).view(1, -1).to(DEVICE)),1)
    output_similarity = similarity_dis.repeat(tgt_out.size()[0], 1)
    output_saliency = saliency_dis.repeat(tgt_out.size()[0], 1)
    output_rgb = rgb_dis.repeat(tgt_out.size()[0], 1)
    for i in range(iter):
        output_similarity_dis = np.random.choice(xk,max_length,p=output_similarity[0,:].numpy())
        output_saliency_dis = np.random.choice(xk,max_length,p=output_saliency[0,:].numpy())
        output_rgb_dis = np.random.choice(xk,max_length,p=output_rgb[0,:].numpy())
        GAZE_similarity[:,i] = output_similarity_dis
        GAZE_saliency[:,i] = output_saliency_dis
        GAZE_rgb[:,i]= output_rgb_dis
    all_gaze_similarity = pd.concat([all_gaze_similarity, pd.DataFrame(GAZE_similarity)],axis=0)
    all_gaze_saliency = pd.concat([all_gaze_saliency, pd.DataFrame(GAZE_saliency)],axis=0)
    all_gaze_rgb = pd.concat([all_gaze_rgb, pd.DataFrame(GAZE_rgb)],axis=0)
    
    loss1 = loss_fn(output_similarity, tgt_out.reshape(-1))
    loss2 = loss_fn(output_saliency, tgt_out.reshape(-1))
    loss3 = loss_fn(output_rgb, tgt_out.reshape(-1))

    losses1 += loss1.item()
    losses2 += loss2.item()
    losses3 += loss3.item()
print('similarity loss:',losses1 / time)
print('saliency loss:',losses2 / time)
print('rgb loss:',losses3 / time)

all_gaze_similarity.to_csv('./dataset/outputdata/gaze_similarity_new.csv', index=False)
all_gaze_saliency.to_csv('./dataset/outputdata/gaze_saliency_new.csv', index=False)
all_gaze_rgb.to_csv('./dataset/outputdata/gaze_rgb_new.csv', index=False)