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
    
    similarity_dis = torch.cat((similarity_dis,torch.tensor([end_pro]).view(tgt_out.size()[1],-1)),1)
    saliency_dis = torch.cat((saliency_dis,torch.tensor([end_pro]).view(tgt_out.size()[1],-1)),1)
    
    output_similarity = torch.zeros((tgt_out.size()[0], tgt_out.size()[1], 31))
    output_saliency = torch.zeros((tgt_out.size()[0], tgt_out.size()[1], 31))
    xk = torch.arange(27)
    xk = torch.cat((xk,torch.tensor([EOS_IDX])))
    mydist1 = rv_discrete(values=(xk.numpy(), similarity_dis.squeeze(0).detach().cpu().numpy()))
    mydist2 = rv_discrete(values=(xk.numpy(), saliency_dis.squeeze(0).detach().cpu().numpy()))
   
    for i in range(tgt_out.size()[0]):
        for j in range(tgt_out.size()[1]):
            data1 = mydist1.rvs(size=1)
            data2 = mydist2.rvs(size=1)
         
            output_similarity[i][j][data1] = 1
            output_saliency[i][j][data2] = 1
 
    loss1 = loss_fn(output_similarity.reshape(-1, output_similarity.shape[-1]), tgt_out.reshape(-1))
    loss2 = loss_fn(output_saliency.reshape(-1, output_saliency.shape[-1]), tgt_out.reshape(-1))

    losses1 += loss1.item()
    losses2 += loss2.item()
print(losses1 / time)
print(losses2 / time)
