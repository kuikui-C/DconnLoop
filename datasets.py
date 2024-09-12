import os
import h5py
import os.path as osp
import numpy as np
import random
from sklearn.model_selection import train_test_split
import torch
from torch.utils import data

__all__ = ['EPIDataSetTrain', 'EPIDataSetTest']


class EPIDataSetTrain(data.Dataset):
    def __init__(self, data_tr, label_tr):
        super(EPIDataSetTrain, self).__init__()
        self.data = data_tr
        self.label = label_tr
        assert len(self.data) == len(self.label), \
            "the number of sequences and labels must be consistent."

        print("The number of train positive data is {}".format(sum(self.label.reshape(-1) == 1)))
        print("The number of train negative data is {}".format(sum(self.label.reshape(-1) == 0)))
        print("pre-process data is done.")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_one = self.data[index]
        label_one = self.label[index]

        return {"data": data_one, "label": label_one}


class EPIDataSetTest(data.Dataset):
    def __init__(self, data_te, label_te):
        super(EPIDataSetTest, self).__init__()
        self.data = data_te
        self.label = label_te

        # if len(self.data) != len(self.label):
        #     min_length = min(len(self.data), len(self.label))
        #     self.data = self.data[:min_length]
        #     self.label = self.label[:min_length]

        assert len(self.data) == len(self.label), \
            "the number of sequences and labels must be consistent."
        print("The number of test positive data is {}".format(sum(self.label.reshape(-1) == 1)))
        print("The number of test negative data is {}".format(sum(self.label.reshape(-1) == 0)))
        print("pre-process data is done.")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_one = self.data[index]
        label_one = self.label[index]

        return {"data": data_one, "label": label_one}

class EPIDataSetVal(data.Dataset):
    def __init__(self, data_va, label_va):
        super(EPIDataSetVal, self).__init__()
        self.data = data_va
        self.label = label_va

        # if len(self.data) != len(self.label):
        #     min_length = min(len(self.data), len(self.label))
        #     self.data = self.data[:min_length]
        #     self.label = self.label[:min_length]

        assert len(self.data) == len(self.label), \
            "the number of sequences and labels must be consistent."
        print("The number of val positive data is {}".format(sum(self.label.reshape(-1) == 1)))
        print("The number of val negative data is {}".format(sum(self.label.reshape(-1) == 0)))
        print("pre-process data is done.")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_one = self.data[index]
        label_one = self.label[index]

        return {"data": data_one, "label": label_one}
