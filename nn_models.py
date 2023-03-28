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

        return X, y
    

class RNN(torch.nn.Module):
    def __init__(self, hidden_size, num_layers, h0=None):
        super().__init__()
        input_size = 3
        output_size = 4

        self.rnn = torch.nn.RNN(input_size, hidden_size, num_layers, dtype=DTYPE)
        if h0 == None:
            self.h0 = torch.nn.Parameter(torch.randn(num_layers, hidden_size, dtype=DTYPE))
        else:
            assert h0.shape == (num_layers, hidden_size)
            assert h0.dtype == DTYPE
            self.h0 = torch.nn.Parameter(h0)
        self.linear = torch.nn.Linear(hidden_size, output_size, dtype=DTYPE)

        self.to(DEVICE)

    def forward(self, x):
        output, hn = self.rnn(x, self.h0)
        output = self.linear(output)
        return output
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            return output[:, :3]
    

def train_model(model, training_loader,
                epochs = 10,
                verbose = 1,
                ):
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    avg_losses = []
    avg_precision_scores = []
    
    for epoch in range(epochs):  # loop over the dataset multiple times
        if verbose == 1:
            print(f'epoch {epoch}')
            train_iter = tqdm(training_loader)
        else:
            train_iter = iter(training_loader)

        avg_loss = 0
        avg_score = 0

        for data in train_iter:
            model.train()

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.reshape((inputs.shape[1], -1)).to(DEVICE)
            labels = labels.reshape((labels.shape[1], -1)).to(DEVICE)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # record outputs
            avg_loss += loss.item()
            predictions = model.predict(inputs).cpu()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                avg_score += average_precision_score(labels.cpu(), predictions)
            
        avg_loss /= len(training_loader)
        avg_score /= len(training_loader)
        print(f'Loss {avg_loss}')
        print(f'Score {avg_score}')
        avg_losses.append(avg_loss)
        avg_precision_scores.append(avg_score)
    
    return avg_losses, avg_precision_scores

def avg_precision_score(model, data_loader):
    model.eval()

    avg_score = 0
    for data in data_loader:
        inputs, labels = data
        inputs = inputs.reshape((inputs.shape[1], -1)).to(DEVICE)
        labels = labels.reshape((labels.shape[1], -1))[:, :3]

        predictions = model.predict(inputs).cpu()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            avg_score += average_precision_score(labels, predictions)
    return avg_score/len(data_loader)


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    training_data = DefogDataset()
    training_loader = DataLoader(training_data, batch_size=1, shuffle=True)

    model = RNN(4,4)
    model.to(DEVICE)

    train_model(model, training_loader,
                epochs = 2,
                verbose = 1,
                )
