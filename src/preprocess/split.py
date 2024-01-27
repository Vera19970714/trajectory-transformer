import random
import pickle
import numpy as np
file='dataset/processdata/dataset_amazon'
with open(file, "rb") as fp:  # Unpickling
    raw_data = pickle.load(fp)


'''data = []
wine_task = []
for index in range(len(raw_data)):
    if raw_data[index]['id'] == 'Q3':
        data.append(index)
    elif raw_data[index]['id'] == 'Q1':
        wine_task.append(index)
        data.append(index)'''
data = raw_data

# raw_data = torch.load(file)
data_length = len(data)
# Generate the list from 453 to 891
number_list = list(range(0, data_length))

# Shuffle the list randomly
random.shuffle(number_list)
#indices = data[np.array(number_list)]

with open('dataset/processdata/splitlist_all_amazon.txt', 'w') as f:
    f.write('\n'.join(map(str, number_list)))



# Calculate the sizes of each split
'''total_size = len(number_list)
train_size = int(0.8 * total_size)
valid_size = int(0.1 * total_size)

# Split the list into training, validation, and test sets
train_list = number_list[:train_size]
valid_list = number_list[train_size:train_size+valid_size]
test_list = number_list[train_size+valid_size:]

# Save the lists to separate text files
with open('splitlist_combine_only_indices_train.txt', 'w') as f:
    f.write('\n'.join(map(str, train_list)))

with open('splitlist_combine_only_indices_valid.txt', 'w') as f:
    f.write('\n'.join(map(str, valid_list)))'''



