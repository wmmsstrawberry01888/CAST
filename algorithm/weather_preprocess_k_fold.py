import math
import sys

import numpy as np
import pandas
import torch
# from torch.utils.data import Dataset
from algorithm.base_dataset import BaseDataset
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.paramters import get_params

from algorithm.base_dataset import search_recent_data

# class BoeingDataset(Dataset):
class BoeingDataset(BaseDataset):
    def __init__(self, clean_data, data_range, config):
        self.T_h = config.model.T_h
        self.T_p = config.model.T_p
        self.V = config.model.V
        self.points_per_hour = config.data.points_per_hour
        self.data_range = data_range
        # self.data_name = clean_data.data_name
        self.data_name = ""

        self.label = np.array(clean_data) # (T_total, V, D), where T_all means the total time steps in the data
        self.feature = np.array(clean_data)  # (T_total, V, D)

        # Prepare samples
        # self.idx_lst = self.get_idx_lst(self)
        self.idx_lst = self.get_idx_lst()
        print('sample num:', len(self.idx_lst))

    def __getitem__(self, index):

        recent_idx = self.idx_lst[index]

        start, end = recent_idx[1][0], recent_idx[1][1]
        label = self.label[start:end]

        start, end = recent_idx[0][0], recent_idx[0][1]
        node_feature = self.feature[start:end]
        pos_w, pos_d = self.get_time_pos(start)
        pos_w = np.array(pos_w, dtype=np.int32)
        pos_d = np.array(pos_d, dtype=np.int32)
        return label, node_feature, pos_w, pos_d, torch.Tensor([1, 1, 1] * len(label))

    def __len__(self):
        return len(self.idx_lst)

    def get_time_pos(self, idx):
        idx = np.array(range(self.T_h)) + idx
        pos_w = (idx // (self.points_per_hour * 24)) % 7  # day of week
        pos_d = idx % (self.points_per_hour * 24)  # time of day
        return pos_w, pos_d


    # def get_idx_lst(self):
    #     idx_lst = []
    #     start = self.data_range[0]
    #     end = self.data_range[1] if self.data_range[1] != -1 else self.feature.shape[0]
    #
    #     for label_start_idx in range(start, end):
    #         # only 6:00-24:00 for Metro data
    #         if 'Metro' in self.data_name:
    #             if label_start_idx % (24 * 6) < (7 * 6):
    #                 continue
    #             if label_start_idx % (24 * 6) > (24 * 6) - self.T_p:
    #                 continue
    #
    #         recent = search_recent_data(self.feature, label_start_idx, self.T_p, self.T_h)  # recent data
    #
    #         if recent:
    #             idx_lst.append(recent)
    #     return idx_lst


def key2value(data, key2value_map):
    new_data = []
    for i in range(len(data)):
        new_data.append(float(key2value_map[data[i]]))
    return new_data


class DataPreprocess:
    def __init__(self, file_path, max_node_dim, node_index_list):
        self.data = self.read_data(file_path, max_node_dim, node_index_list)
        self.data = self.normalization(self.data).astype('float32')

    def get_data(self):
        return self.data

    @staticmethod
    def read_data(file_path, max_node_dim, node_index_list):
        """
        max_node_dim:
        node_index:
        """

        data = pandas.read_csv(file_path)
        max_data_len = len(data)
        final_data = []
        mask_value = 0.
        str_key = ["Install.Type", "Borough", "ntacode"]
        set_1 = set(data[str_key[0]])
        map_1 = {}
        count = 0
        for tem_set in set_1:
            map_1[tem_set] = count
            count += 1
        set_2 = set(data[str_key[1]])
        map_2 = {}
        count = 0
        for tem_set in set_2:
            map_2[tem_set] = count
            count += 1
        set_3 = set(data[str_key[2]])
        map_3 = {}
        count = 0
        for tem_set in set_3:
            map_3[tem_set] = count
            count += 1
        for node_index in node_index_list:
            tem_data = []
            for i in range(max_node_dim):
                if i < len(node_index):
                    if node_index[i] == "Day":
                        timestamp = data[node_index[i]].values.tolist()
                        time_feature1 = []
                        time_feature2 = []
                        time_feature3 = []
                        for j in range(len(timestamp)):
                            time_str = timestamp[j]
                            time_values = time_str.split('/')
                            # time_feature1.append(int(time_values[0]) + int(time_values[1]))
                            # time_feature2.append(int(time_values[2]))
                            # time_feature3.append(int(time_values[3])//62)
                            time_feature1.append(int(time_values[0]))
                            time_feature2.append(int(time_values[1]))
                            time_feature3.append(int(time_values[2]))
                        tem_data.append(time_feature1)
                        tem_data.append(time_feature2)
                        tem_data.append(time_feature3)
                        break
                    else:
                        if node_index[i] == str_key[0]:
                            new_data = key2value(data[node_index[i]].values.tolist(), map_1)
                            tem_data.append(new_data)
                        elif node_index[i] == str_key[1]:
                            new_data = key2value(data[node_index[i]].values.tolist(), map_2)
                            tem_data.append(new_data)
                        elif node_index[i] == str_key[2]:
                            new_data = key2value(data[node_index[i]].values.tolist(), map_3)
                            tem_data.append(new_data)
                        else:
                            tem_data.append(data[node_index[i]].values.tolist())
                else:
                    tem_data.append([mask_value] * max_data_len)
            final_data.append(tem_data)
        data_array = np.array(final_data)
        data_array = data_array.transpose(2, 0, 1)
        return data_array.tolist()

    def normalization(self, feature):
        feature_array = np.array(feature)
        _ , node_num, feature_num = np.shape(feature)
        self.mean = [[0 for i in range(feature_num)] for j in range(node_num)]
        self.std = [[0 for i in range(feature_num)] for j in range(node_num)]
        for i in range(node_num):
            for j in range(feature_num):
                feature_array[:, i, j] = self.normalization_one_feature(feature_array[:, i, j], i, j)
        return feature_array

    def normalization_one_feature(self, feature, node_index, feature_index):
        # train = feature[:self.val_start_idx]
        train = np.nan_to_num(feature)
        # if 'Metro' in self.data_name:
        #     idx_lst = [i for i in range(train.shape[0]) if i % (24 * 6) >= 7 * 6 - 12]
        #     train = train[idx_lst]

        mean = np.mean(train)
        std = np.std(train)

        # since the feature is actual the flow, the mean and std of feature is also the label's mean and std
        self.mean[node_index][feature_index] = mean
        self.std[node_index][feature_index] = std
        if std == 0:
            return feature
        return (feature - mean) / std


    def reverse_normalization(self, x):
        x = np.array(x)
        _, _, node_num, feature_num = np.shape(x)
        for i in range(node_num):
            for j in range(feature_num):
                x[:, :, i, j] = self.reverse_normalization_one_feature(x[:, :, i, j], i, j)
        return x

    def reverse_normalization_one_feature(self, x, node_index, feature_index):
        return self.mean[node_index][feature_index] + self.std[node_index][feature_index] * x

# if __name__ == "__main__":