import math
import sys

import numpy as np
import pandas
import torch
from torch.utils.data import Dataset
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.paramters import get_params

from algorithm.base_dataset import search_recent_data


class BoeingDataset(Dataset):
    """Sliding-window dataset for ETTh1.

    Mirrors the exchange_rate preprocessor: samples are (T_h history, T_p
    forecast) windows over a (T_total, V, D) array. The same interface is used
    by CAST and every baseline, so all methods are evaluated through one
    identical pipeline.
    """

    def __init__(self, clean_data, data_range, config):
        self.T_h = config.model.T_h
        self.T_p = config.model.T_p
        self.V = config.model.V
        self.points_per_hour = config.data.points_per_hour
        self.data_range = data_range
        self.data_name = ""

        self.label = np.array(clean_data)    # (T_total, V, D)
        self.feature = np.array(clean_data)  # (T_total, V, D)

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
        pos_d = idx % (self.points_per_hour * 24)         # time of day
        return pos_w, pos_d

    def get_idx_lst(self):
        idx_lst = []
        start = self.data_range[0]
        end = self.data_range[1] if self.data_range[1] != -1 else self.feature.shape[0]

        for label_start_idx in range(start, end):
            recent = search_recent_data(self.feature, label_start_idx, self.T_p, self.T_h)
            if recent:
                idx_lst.append(recent)
        return idx_lst

class DataPreprocess:
    """ETTh1 loader with per-channel (per node, per feature) z-score.

    Each (node, feature) channel is standardized on its own mean/std, exactly
    like the exchange_rate / weather / pci preprocessors, so variables with
    different physical units are each normalized on their own scale. The same
    normalization is applied to CAST and all baselines.
    """

    def __init__(self, file_path, max_node_dim, node_index_list):
        self.data = self.read_data(file_path, max_node_dim, node_index_list)
        self.data = self.normalization(self.data).astype('float32')

    def get_data(self):
        return self.data

    @staticmethod
    def read_data(file_path, max_node_dim, node_index_list):
        """Read the ETTh1 CSV and pack it into a (T_total, V, D) array.

        node_index_list groups the CSV columns into V nodes, each holding up to
        max_node_dim (=D) feature columns. The 'date' column is expanded into 3
        integer time features (month, day, hour) that occupy node 0; short
        groups are zero-padded to D. This matches the layout consumed by the
        shared dataloader (channels 0:3 are the date features skipped in
        metrics, see engine/evaluator.py).
        """
        data = pandas.read_csv(file_path)
        max_data_len = len(data)
        final_data = []
        mask_value = 0.
        for node_index in node_index_list:
            tem_data = []
            for i in range(max_node_dim):
                if i < len(node_index):
                    if node_index[i] == "date":
                        timestamp = data[node_index[i]].values.tolist()
                        time_feature1 = []  # month
                        time_feature2 = []  # day
                        time_feature3 = []  # hour
                        for j in range(len(timestamp)):
                            # ETTh1 timestamps look like '2016-07-01 00:00:00'
                            time_str = str(timestamp[j])
                            date_part, _, time_part = time_str.partition(' ')
                            ymd = date_part.split('-')
                            hour = time_part.split(':')[0] if time_part else '0'
                            time_feature1.append(int(ymd[1]))          # month
                            time_feature2.append(int(ymd[2]))          # day
                            time_feature3.append(int(hour))            # hour
                        tem_data.append(time_feature1)
                        tem_data.append(time_feature2)
                        tem_data.append(time_feature3)
                        break
                    else:
                        tem_data.append(data[node_index[i]].values.tolist())
                else:
                    tem_data.append([mask_value] * max_data_len)
            final_data.append(tem_data)
        data_array = np.array(final_data)
        data_array = data_array.transpose(2, 0, 1)  # (T_total, V, D)
        return data_array.tolist()

    def normalization(self, feature):
        feature_array = np.array(feature)
        _, node_num, feature_num = np.shape(feature)
        self.mean = [[0 for _ in range(feature_num)] for _ in range(node_num)]
        self.std = [[0 for _ in range(feature_num)] for _ in range(node_num)]
        for i in range(node_num):
            for j in range(feature_num):
                feature_array[:, i, j] = self.normalization_one_feature(
                    feature_array[:, i, j], i, j)
        return feature_array

    def normalization_one_feature(self, feature, node_index, feature_index):
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
                x[:, :, i, j] = self.reverse_normalization_one_feature(x[:, :, i, j], i, j)
        return x

    def reverse_normalization_one_feature(self, x, node_index, feature_index):
        return self.mean[node_index][feature_index] + self.std[node_index][feature_index] * x

