from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import torch
import os

path = "dataset/csv"
cry = "cry"
noise = "noise"


class DataCry(Dataset):
    def __init__(self):
        self.fullpath = os.path.join(path, cry)
        self.listpath = os.listdir(self.fullpath)

    def __getitem__(self, item):
        index = self.listpath[item]
        item_path = os.path.join(self.fullpath, index)
        x = np.loadtxt(item_path, delimiter=',')
        x = torch.tensor(torch.from_numpy(x), dtype=torch.float)
        x = torch.reshape(x, (1, 199, 13))
        y = torch.tensor([1],dtype=torch.float)
        return x, y

    def __len__(self):
        return len(self.listpath)


class DataNoise(Dataset):
    def __init__(self):
        self.fullpath = os.path.join(path, noise)
        self.listpath = os.listdir(self.fullpath)

    def __getitem__(self, item):
        index = self.listpath[item]
        item_path = os.path.join(self.fullpath, index)
        x = np.loadtxt(item_path, delimiter=',')
        x = torch.tensor(torch.from_numpy(x), dtype=torch.float)
        x = torch.reshape(x, (1, 199, 13))
        y = torch.tensor([0],dtype=torch.float)
        return x, y

    def __len__(self):
        return len(self.listpath)
