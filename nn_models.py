import numpy as np 
import pandas as pd 
import seaborn as sns
import os
import matplotlib.pyplot as plt
import yaml
from torch.utils.data import Dataset
import torch

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)


class DefogDataset(Dataset):
    def __init__(self):
        self.path = os.path.join(config['data_path'], 'train', 'defog')
        self.ids = os.listdir(self.path)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_path = os.path.join(self.path, self.ids[idx])
        df = pd.read_csv(id_path)

        X = torch.tensor(df[['AccV', 'AccML', 'AccAP']].values)
        y = torch.tensor(df[['StartHesitation']].values)

        return X, y