import numpy as np
import math
import pandas as pd

data_csv = pd.read_csv("dataset.csv")
data = data_csv.to_numpy()
inputpartition = "inputpartition.txt"
partition = {}
max_gain = -10
max_gain_index = None
max_partition_id = None
attributes = data.shape[1]
with open(inputpartition) as f:
    for line in f.readlines():
        word = line.split(' ')
        key = word[0]
        value = []
        for i in range(1, len(word)):
            value.append(int(word[i]))
        partition[key] = value


def calculate_entropy(data, feature_index):
    total_samples = data.shape[0]
    value_dict = {}   
    for index, value in enumerate(data[:, feature_index]):
        target_variable = data[index][data.shape[1] - 1]
        if value in value_dict.keys():
            if target_variable in value_dict[value].keys():
                value_dict[value][target_variable] += 1
            else:
                value_dict[value][target_variable] = 1
        else:
            value_dict[value] = {target_variable: 1}
        if "others" not in value_dict[value].keys():
            value_dict[value]["others"] = 1
        else:
            value_dict[value]["others"] += 1
    total_entropy = 0
    for value in value_dict.keys():
        entropy = 0
        if feature_index != data.shape[1] - 1:
            total_targets = value_dict[value]["others"]
        else:
            total_targets = total_samples
        for target in value_dict[value].keys():
            if target != "others":
                probability = value_dict[value][target] / total_targets
                entropy += -probability * math.log(probability, 2)
        total_entropy += (total_targets / total_samples) * entropy
    return total_entropy


def get_partitions(data, feature_index, partition_id):
    partitions = {}
    for index, value in zip(partition[partition_id], data[np.array(partition[partition_id]) - 1, feature_index]):
        part_id = partition_id + str(value)
        if part_id in partitions.keys():
            partitions[part_id].append(index)
        else:
            partitions[part_id] = [index]

    return partitions



for partition_id in partition.keys():
    data_part = data[np.array(partition[partition_id]) - 1]
    entropy_S = calculate_entropy(data_part, attributes - 1)
    F_score = -10
    for i in range(attributes - 1):
        gain = entropy_S - calculate_entropy(data_part, i)
        if F_score < gain:
            F_score = gain
            max_gain_index_part = i
    F_score = (data_part.shape[0] / data.shape[0]) * F_score
    if max_gain <= F_score:
        max_gain = F_score
        max_gain_index = max_gain_index_part
        max_partition_id = partition_id

new_partitions = get_partitions(data, max_gain_index, max_partition_id)
print(new_partitions)
print("The partition with maximum Gain is", max_partition_id ,"with new rows as" , new_partitions.keys(), "with entropy of", max_gain)