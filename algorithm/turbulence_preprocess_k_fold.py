"""Turbulence dataset preprocessing (k-fold variant)."""
import os
import sys

import numpy as np
import pandas
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.paramters import get_params
from algorithm.base_dataset import BaseDataset, search_recent_data


class BoeingDataset(BaseDataset):
    def __init__(self, clean_data, data_range, config):
        super().__init__(clean_data, data_range, config)

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


class DataPreprocess:
    def __init__(self, file_path, max_node_dim, node_index_list):
        self.data = self.read_data(file_path, max_node_dim, node_index_list)
        self.data = self.normalization(self.data).astype('float32')

    def get_data(self):
        return self.data

    @staticmethod
    def read_data(file_path, max_node_dim, node_index_list):
        data = pandas.read_csv(file_path)
        max_data_len = len(data)
        final_data = []
        mask_value = 0.
        for node_index in node_index_list:
            tem_data = []
            for i in range(max_node_dim):
                if i < len(node_index):
                    tem_data.append(data[node_index[i]].values.tolist())
                else:
                    tem_data.append([mask_value] * max_data_len)
            final_data.append(tem_data)
        data_array = np.array(final_data)
        data_array = data_array.transpose(2, 0, 1)
        return data_array.tolist()

    def normalization(self, feature):
        feature_array = np.array(feature)
        _, node_num, feature_num = np.shape(feature)
        self.mean = [[0 for _ in range(feature_num)] for _ in range(node_num)]
        self.std = [[0 for _ in range(feature_num)] for _ in range(node_num)]
        for i in range(node_num):
            for j in range(feature_num):
                feature_array[:, i, j] = self._norm_one(feature_array[:, i, j], i, j)
        return feature_array

    def _norm_one(self, feature, node_index, feature_index):
        train = np.nan_to_num(feature)
        mean = np.mean(train)
        std = np.std(train)
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
                x[:, :, i, j] = self.mean[i][j] + self.std[i][j] * x[:, :, i, j]
        return x
