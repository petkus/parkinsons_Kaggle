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
from data_pipeline import DefogDataset


DTYPE = torch.float
if torch.cuda.is_available():
    DEVICE = torch.device('cuda') 
else:
    DEVICE = torch.device('cpu') 

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)
    

class RNN(torch.nn.Module):
    """
        RNN
    """
    def __init__(self, hidden_size, num_layers, h0=None, bidirectional=False):
        super().__init__()
        input_size = 3
        output_size = 4

        self.rnn = torch.nn.RNN(input_size, hidden_size, num_layers, dtype=DTYPE, bidirectional=bidirectional)

        if bidirectional:
            D = 2
        else:
            D = 1
        if h0 == None:
            self.h0 = torch.nn.Parameter(torch.randn(D*num_layers, hidden_size, dtype=DTYPE))
        else:
            assert h0.shape == (num_layers, hidden_size)
            assert h0.dtype == DTYPE
            self.h0 = torch.nn.Parameter(h0)
        self.linear = torch.nn.Linear(D*hidden_size, output_size, dtype=DTYPE)

        self.to(DEVICE)

        self.to(DEVICE)

    def forward(self, x):
        output, hn = self.rnn(x, self.h0)
        output = self.linear(output)
        return output
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            return torch.sigmoid(output[:, :3])


def train_model(model, training_loader,
                validation_loader = None,
                epochs = 10,
                verbose = 1,
                ):
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    output = {}
    output['avg_losses'] = []
    output['avg_precision_scores'] = []

    if validation_loader != None:
        output['val_losses'] = []
        output['val_avg_precision_scores'] = []
    
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
        output['avg_losses'].append(avg_loss)
        output['avg_precision_scores'].append(avg_score)
        print(f'Training loss {avg_loss}')
        print(f'Training score {avg_score}')

        model.eval()
        if validation_loader != None:
            avg_loss = 0
            for data in validation_loader:
                inputs, labels = data
                inputs = inputs.reshape((inputs.shape[1], -1)).to(DEVICE)
                labels = labels.reshape((labels.shape[1], -1)).to(DEVICE)
                avg_loss += criterion(outputs, labels).item()
            output['val_losses'].append(avg_loss)
            output['val_avg_precision_scores'].append(score_model(model, validation_loader))

    
    return output

def score_model(model, data_loader):
    """
        Gets the average precision score of model applied to dataloader
    """
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
