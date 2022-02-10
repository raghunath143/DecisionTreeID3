import numpy as np
import math

# data = np.array([	[0, 0, 0, 0],
# 					[0, 0, 1, 0],
# 					[1, 0, 2, 0],
# 					[2, 2, 0, 1],
# 					[0, 0, 1, 1],
# 					[0, 1, 2, 1],
# 					[1, 1, 1, 1],
# 					[1, 2, 2, 0],
# 					[0, 0, 1, 0],
# 					[2, 0, 1, 0]])

# partition = {'x': [1, 10], 'y2': [2, 3, 4, 5], 'z': [6, 7, 8, 9]}

data = np.array([ 	
				['R', 'Y', '0', 'a'],
				['R', 'X', '0', 'a'],
				['R', 'X', '0', 'b'],
				['R', 'X', '1', 'a'],
				['G', 'X', '1', 'b'],
				['G', 'X', '1', 'a'],
				['G', 'X', '1', 'a'],
				['G', 'X', '1', 'b'],
				['B', 'Y', '1', 'a'],
				['B', 'Y', '0', 'b'],
				], dtype = 'str')

partition = {'s1':[1, 2, 3, 4], 's2':[5, 6, 7, 8], 's3': [9, 10]}


def calculate_entropy(data, feature_index):
	value_dict = {}
	total_samples = data.shape[0]
	for idx, value in enumerate(data[:, feature_index]):
		target_var = data[idx][data.shape[1]-1]
		if value in value_dict.keys():
			if target_var in value_dict[value].keys():
				value_dict[value][target_var] += 1
			else:
				value_dict[value][target_var] = 1
				
		else:
			value_dict[value] = {target_var:1}
		if "total" not in value_dict[value].keys():
			value_dict[value]["total"] = 1
		else:
			value_dict[value]["total"] += 1
	
	total_entropy = 0
	for value in value_dict.keys():
		entropy = 0
		if feature_index != data.shape[1]-1:
			total_targets = value_dict[value]["total"]
		else:
			total_targets = total_samples
		for target in value_dict[value].keys():
			if target != "total":
				prob = value_dict[value][target] / total_targets
				entropy += -prob * math.log(prob, 2)
		total_entropy += (total_targets/total_samples) * entropy					
	#print("feature index:", feature_index, "value dict:", value_dict, "entropy:", total_entropy)
	return total_entropy

def get_partitions(data, feature_index, partition_id):
	partitions = {}
	for idx, value in zip(partition[partition_id], data[np.array(partition[partition_id])-1, feature_index]):
		new_id = partition_id + str(value)
		if new_id not in partitions.keys():
			partitions[new_id] = [idx]
		else:
			partitions[new_id].append(idx)
	return partitions

max_gain = -10
max_gain_idx = None
max_part_id = None
attributes = data.shape[1]
for partition_id in partition.keys():
	data_p = data[np.array(partition[partition_id])-1]
	entropy_S = calculate_entropy(data_p, attributes-1)
	max_gain_p = -10
	for i in range(attributes-1):
		gain = entropy_S - calculate_entropy(data_p, i)
		if max_gain_p < gain:
			max_gain_p = gain
			max_gain_idx_p = i
	max_gain_p = (data_p.shape[0]/data.shape[0]) * max_gain_p
	# print("F score:", max_gain_p)
	if max_gain <= max_gain_p:
		max_gain = max_gain_p
		max_gain_idx = max_gain_idx_p
		max_part_id = partition_id

new_partitions = get_partitions(data, max_gain_idx, max_part_id)
print("new partitions:", new_partitions)
print("The partition, feature with maximum Gain is", max_part_id, max_gain_idx, "with entropy of", max_gain)