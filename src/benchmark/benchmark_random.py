import numpy as np
from random import randint
import pandas as pd
# seed amazon_random number generator
#seed(1)

def sample_gaze_from_distri(avg_len, distri, TOTAL_PCK):
    end_prob = 1 / avg_len * 10000
    gaze = []
    x = randint(0, 10000)
    minLen = 1
    while x >= end_prob or len(gaze) < minLen:
        ind = np.random.choice(TOTAL_PCK, 1, p=distri)
        gaze.append(np.ndarray.item(ind))
        x = randint(0, 10000)
    gaze = np.stack(gaze).reshape(1, -1)
    return gaze

# TODO: carefully fit the number accordingly
iter = 100
avg_len = 8.35
TOTAL_PCK = 84
minLen = 1
#test_datapath = '../dataset/processdata/dataset_Q23_mousedel_time_val'
all_gaze = pd.DataFrame()
datalength = 86
for i in range(datalength):
    for n in range(iter):
        GAZE = sample_gaze_from_distri(avg_len,np.ones(TOTAL_PCK) / TOTAL_PCK, TOTAL_PCK)
        gaze_df = np.stack(GAZE).reshape(1, -1)
        all_gaze = pd.concat([all_gaze, pd.DataFrame(gaze_df)], axis=0)


loss = np.log(TOTAL_PCK+3)
print('loss=', loss)
all_gaze.to_csv('./dataset/checkEvaluation/amazon_random/gaze_expect.csv', index=False)

