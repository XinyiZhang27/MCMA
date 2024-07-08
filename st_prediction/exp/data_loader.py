import os
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from myutils.tools import check_and_create_path


import warnings
warnings.filterwarnings('ignore')


class Dataset_Custom(Dataset):
    def __init__(self, data_path, scaler_path, scaler, flag='train'):
        self.data_path = data_path
        self.scaler_path = scaler_path
        self.scaler = scaler

        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.__read_data__()

    def __read_data__(self):
        all_data = np.load(self.data_path, allow_pickle=True)
        self.seqs_x = all_data["seqs_x"]
        self.seqs_x_mark = all_data["seqs_x_mark"]
        self.seqs_y = all_data["seqs_y"]
        self.seqs_y_mark = all_data["seqs_y_mark"]

        # scale
        if self.set_type == 0:
            self.scaler.fit(self.seqs_x)
            check_and_create_path(self.scaler_path)
            np.savez(self.scaler_path, mean=self.scaler.mean, std=self.scaler.std)
        self.seqs_x = self.scaler.transform(self.seqs_x)

    def __getitem__(self, index):
        return self.seqs_x[index], self.seqs_y[index], self.seqs_x_mark[index], self.seqs_y_mark[index]

    def __len__(self):
        return self.seqs_x.shape[0]

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, scaler, seqs_x, seqs_x_mark, seqs_y, seqs_y_mark):
        self.seqs_x = seqs_x
        self.seqs_x_mark = seqs_x_mark
        self.seqs_y = seqs_y
        self.seqs_y_mark = seqs_y_mark

        self.scaler = scaler
        self.seqs_x = self.scaler.transform(self.seqs_x)

    def __getitem__(self, index):
        return self.seqs_x[index], self.seqs_y[index], self.seqs_x_mark[index], self.seqs_y_mark[index]

    def __len__(self):
        return self.seqs_x.shape[0]

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
