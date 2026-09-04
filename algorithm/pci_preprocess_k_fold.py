import math
import sys

import numpy as np
import pandas
import torch
from torch.utils.data import Dataset
from algorithm.base_dataset import BaseDataset
import os
import time
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.paramters import get_params

from algorithm.base_dataset import search_recent_data


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
        for node_index in node_index_list:
            tem_data = []
            time_total_col = max_node_dim
            for i in tqdm(range(max_node_dim)):
                if i < len(node_index):
                    if node_index[i] == "Datetimes":
                        timestamp = data[node_index[i]].values.tolist()
                        time_feature1 = []
                        time_feature2 = []
                        time_feature3 = []
                        time_feature4 = []
                        time_feature5 = []
                        time_feature6 = []
                        for j in range(len(timestamp)):
                            time_str = timestamp[j]
                            time_array = time.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                            time_feature1.append(time_array.tm_year)
                            time_feature2.append(time_array.tm_mon)
                            time_feature3.append(time_array.tm_mday)
                            time_feature4.append(time_array.tm_hour)
                            time_feature5.append(time_array.tm_min)
                            time_feature6.append(time_array.tm_sec)
                        tem_data.append(time_feature1)
                        tem_data.append(time_feature2)
                        tem_data.append(time_feature3)
                        tem_data.append(time_feature4)
                        tem_data.append(time_feature5)
                        tem_data.append(time_feature6)
                        time_total_col -= 6
                    else:
                        tem_data.append(data[node_index[i]].values.tolist())
                        time_total_col -= 1
                else:
                    if time_total_col > 0:
                        tem_data.append([mask_value] * max_data_len)
                        time_total_col -= 1


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

if __name__ == "__main__":
#     # file_path = "/home/tureful/
    file_path = "../data/pci/DMSP_final_training.csv"
    node_index_list = [["Datetimes", "SC_AACGM_LAT", "ELE_TOTAL_ENERGY_FLUX", "ID_SC"],
                       ["sin_ut", "cos_ut", "sin_doy", "cos_doy", "sin_SC_AACGM_LTIME", "cos_SC_AACGM_LTIME"],
                       ["F107", "Bz", "By", "Bx", "AL", "AU", "SymH", "PC", "vsw", "psw"],
                       ["Bz_6hr", "By_6hr", "Bx_6hr", "AL_6hr", "AU_6hr", "SymH_6hr", "PC_6hr", "vsw_6hr", "psw_6hr"],
                       ["Bz_3hr", "By_3hr", "Bx_3hr", "AL_3hr", "AU_3hr", "SymH_3hr", "PC_3hr", "vsw_3hr", "psw_3hr"],
                       ["Bz_1hr", "By_1hr", "Bx_1hr", "AL_1hr", "AU_1hr", "SymH_1hr", "PC_1hr", "vsw_1hr", "psw_1hr"],
                       ["Bz_45min", "By_45min", "Bx_45min", "AL_45min", "AU_45min", "SymH_45min", "PC_45min", "vsw_45min", "psw_45min"],
                       ["Bz_30min", "By_30min", "Bx_30min", "AL_30min", "AU_30min", "SymH_30min", "PC_30min", "vsw_30min", "psw_30min"],
                       ["Bz_10min", "By_10min", "Bx_10min", "AL_10min", "AU_10min", "SymH_10min", "PC_10min", "vsw_10min", "psw_10min"]]
#
    config = get_params()
#
    air_data = DataPreprocess(file_path, 10, node_index_list)
    print(np.shape(air_data))
    construct_data = air_data.get_data()
    dataset = BoeingDataset(construct_data, [0, 200000], config)
    print(dataset.__getitem__(0))
