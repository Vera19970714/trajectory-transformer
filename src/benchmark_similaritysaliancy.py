import numpy as np
from torch.utils.data import DataLoader
from data_builder import SimilarityFixDataset, collate_fn_similarity
import torch
from scipy.stats import rv_discrete

# seed random number generator
#seed(1)
avg_len = 7.18
end_pro = 0.14
continue_pro = 0.86

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 27, 28, 29, 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_datapath = './dataset/processdata/dataset_Q23_similarity_time_val'
test_set = SimilarityFixDataset(0, test_datapath)
test_loader = DataLoader(dataset=test_set, batch_size=1, num_workers=0, collate_fn=collate_fn_similarity, shuffle=False)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
losses1 = 0
losses2 =0
time = 0

for package_similarity, package_saliency, package_seq in test_loader:
    time += 1
    package_similarity = package_similarity.to(DEVICE)
    package_saliency = package_saliency.to(DEVICE)
    package_seq = package_seq.to(DEVICE)

    tgt_out = package_seq[1:, :]

    similarity_dis = (continue_pro/torch.sum(package_similarity))*package_similarity
    saliency_dis = (continue_pro/torch.sum(package_saliency))*package_saliency

    # 1, 28
    similarity_dis = torch.cat((similarity_dis,torch.tensor([0, 0, 0, end_pro]).view(1, -1).to(DEVICE)),1)
    saliency_dis = torch.cat((saliency_dis,torch.tensor([0, 0, 0, end_pro]).view(1, -1).to(DEVICE)),1)

    output_similarity = similarity_dis.repeat(tgt_out.size()[0], 1)
    output_saliency = saliency_dis.repeat(tgt_out.size()[0], 1)
 
    loss1 = loss_fn(output_similarity, tgt_out.reshape(-1))
    loss2 = loss_fn(output_saliency, tgt_out.reshape(-1))

    losses1 += loss1.item()
    losses2 += loss2.item()
print('similarity loss:',losses1 / time)
print('saliency loss:',losses2 / time)
