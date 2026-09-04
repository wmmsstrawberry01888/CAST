import sys

import numpy as np
import pandas
import torch
from torch.utils.data import Dataset
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.paramters import get_params


def search_recent_data(train, label_start_idx, T_p, T_h):
    """
    T_p: prediction time steps
    T_h: historical time steps
    """
    if label_start_idx + T_p > len(train): return None
    start_idx, end_idx = label_start_idx - T_h, label_start_idx - T_p + T_p
    if start_idx < 0 or end_idx < 0: return None
    return (start_idx, end_idx), (label_start_idx, label_start_idx + T_p)


class BaseDataset(Dataset):
    def __init__(self, clean_data, data_range, config):
        self.T_h = config.model.T_h
        self.T_p = config.model.T_p
        self.V = config.model.V
        self.points_per_hour = config.data.get('points_per_hour', 12)
        self.data_range = data_range
        self.data_name = getattr(config.data, 'name', '')

        self.label = np.array(clean_data)
        self.feature = np.array(clean_data)

        self.idx_lst = self.get_idx_lst()
        print('sample num:', len(self.idx_lst))

    def __len__(self):
        return len(self.idx_lst)

    def get_time_pos(self, idx):
        idx = np.array(range(self.T_h)) + idx
        pos_w = (idx // (self.points_per_hour * 24)) % 7  # day of week
        pos_d = idx % (self.points_per_hour * 24)  # time of day
        return pos_w, pos_d

    def get_idx_lst(self):
        idx_lst = []
        for tem_data_range in self.data_range:
            start =tem_data_range[0]
            end = tem_data_range[1] if tem_data_range[1] != -1 else self.feature.shape[0]

            for label_start_idx in range(start, end):
                # only 6:00-24:00 for Metro data
                if 'Metro' in self.data_name:
                    if label_start_idx % (24 * 6) < (7 * 6):
                        continue
                    if label_start_idx % (24 * 6) > (24 * 6) - self.T_p:
                        continue

                recent = search_recent_data(self.feature, label_start_idx, self.T_p, self.T_h)  # recent data

                if recent:
                    idx_lst.append(recent)
        return idx_lst