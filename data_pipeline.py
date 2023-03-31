import numpy as np 
import pandas as pd 
import seaborn as sns
import os
import matplotlib.pyplot as plt
import yaml
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
from sklearn.metrics import average_precision_score
import warnings


DTYPE = torch.float
if torch.cuda.is_available():
    DEVICE = torch.device('cuda') 
else:
    DEVICE = torch.device('cpu') 

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

class DefogDataset(Dataset):
    def __init__(self):
        self.path = os.path.join(config['data_path'], 'train', 'tdcsfog')
        self.ids = os.listdir(self.path)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_path = os.path.join(self.path, self.ids[idx])
        df = pd.read_csv(id_path)

        df['None'] = 1 - df[['StartHesitation', 'Turn', 'Walking']].sum(axis=1)

        X = torch.tensor(df[['AccV', 'AccML', 'AccAP']].values, dtype=DTYPE)
        y = torch.tensor(df[['StartHesitation', 'Turn', 'Walking', 'None']].values, dtype=DTYPE)




