import random

# Generate the list from 453 to 891
number_list = list(range(0, 891))

# Shuffle the list randomly
random.shuffle(number_list)

# Calculate the sizes of each split
total_size = len(number_list)
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
    f.write('\n'.join(map(str, valid_list)))

with open('splitlist_combine_only_indices_test.txt', 'w') as f:
    f.write('\n'.join(map(str, test_list)))

