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
        y = torch.tensor(df[['StartHesitation', 'Turn', 'Walking']].values)

        return X, y
    

class RNN(torch.nn.Module):
    def __init__(self, hidden_size, num_layers, h0=None):
        super().__init__()
        input_size = 3
        output_size = 3

        self.rnn = torch.nn.RNN(input_size, hidden_size, num_layers)
        if h0 == None:
            self.h0 = torch.nn.Parameter(torch.randn(num_layers, hidden_size))
        else:
            assert h0.shape == (num_layers, hidden_size)
            self.h0 = torch.nn.Parameter(h0)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, hn = self.rnn(x, self.h0)
        output = self.linear(output)
        return torch.sigmoid(output)