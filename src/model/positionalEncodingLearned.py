import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

num_rows, num_columns = 11, 86
sigma = 2.5
learning_rate = 0.01
num_iterations = 10000
tolerance = 1e-3
# mag of every vector is similar,
# initialized by original PE,
W = nn.Parameter(torch.randn(num_rows, num_columns))

optimizer = torch.optim.Adam([W], lr=learning_rate)


def target_gaussian(i, j, sigma):
    if i == j:
        return torch.tensor(1.0)
    else:
        return torch.exp(-(torch.tensor(i - j, dtype=torch.float32) ** 2) / (2 * sigma ** 2))


for iteration in range(num_iterations):
    optimizer.zero_grad()
    total_loss = 0

    W_normalized = F.normalize(W, p=2, dim=1)

    for i in range(num_rows):
        for j in range(num_rows):
            sim = torch.matmul(W_normalized[i].unsqueeze(0), W_normalized[j].unsqueeze(1))

            target_sim = target_gaussian(i, j, sigma)

            loss = F.mse_loss(sim, target_sim.unsqueeze(0).unsqueeze(1))
            total_loss += loss

    total_loss = total_loss / (num_rows * (num_rows - 1))
    total_loss.backward()
    optimizer.step()

cosine_sim_matrix = torch.zeros((num_rows, num_rows), dtype=torch.float32)
for i in range(num_rows):
    for j in range(num_rows):
        cosine_sim_matrix[i, j] = F.cosine_similarity(W_normalized[i], W_normalized[j], dim=0)

validation_matrix = torch.zeros_like(cosine_sim_matrix, dtype=torch.bool)
for i in range(num_rows):
    for j in range(num_rows):
        target_sim = target_gaussian(i, j, sigma)
        validation_matrix[i, j] = torch.isclose(cosine_sim_matrix[i, j], target_sim, atol=tolerance)

pass_rate = validation_matrix.float().mean().item()
print(f'pass rate: {pass_rate:.2%}')
print('cos_sim_Matrix:', cosine_sim_matrix)
print('Val_Matrix:', validation_matrix)
W = W.detach().cpu().numpy()
np.save('learned_PE.npy', W)