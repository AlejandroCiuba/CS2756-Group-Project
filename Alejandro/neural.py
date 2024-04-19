# Simply neural models for the text experiments
# Alejandro Ciuba, alejandrociuba@pitt.edu
from collections import (OrderedDict,
                         defaultdict, )
from torch.utils.data import (DataLoader,
                              TensorDataset, )

import torch

import numpy as np
import torch.nn as nn


class Neural:

    model: nn.Module | None

    lr: float
    epochs: int
    batch_size: int

    device: str

    loss_record = None
    sampler = None

    loss = None
    optimizer = None

    def __init__(self, **config) -> None:

        self.lr = config['lr']
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.sampler = config['sampler'] if 'sampler' in config else None
        self.loss_record = defaultdict(list)

        self.model = None

    def fit(self, X, y, step_track: int = 10, verbose: bool = True):

        dataset = TensorDataset(X if isinstance(X, torch.Tensor) else torch.from_numpy(X), y if isinstance(y, torch.Tensor) else torch.from_numpy(y))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True if self.sampler is None else False, num_workers=4, sampler=self.sampler)

        n_total_steps = len(dataloader)

        for epoch in range(self.epochs):
            
            for i, (X, y) in enumerate(dataloader):

                X = X.to(self.device)
                y = y.to(self.device)

                outputs = self.model(X)
                loss = self.loss(outputs, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (i + 1) % (n_total_steps // step_track) == 0:

                    if verbose:
                        print('epoch: %d/%d, step: %d/%d, loss=%.4ff' % (epoch+1, self.epochs, i+1, n_total_steps, loss.item()))

                    self.loss_record['epoch'].append(epoch)
                    self.loss_record['step'].append((i + 1) / n_total_steps)
                    self.loss_record['loss'].append(loss.item())

    def predict(self, X) -> tuple[list, list]:

        dataset = TensorDataset(X if isinstance(X, torch.Tensor) else torch.from_numpy(X), torch.zeros(X.shape[0]))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

        with torch.no_grad():

            preds = None

            for X, y in dataloader:

                X = X.to(self.device)
                y = y.to(self.device)
                outputs = self.model(X)

                _, batch_preds = torch.max(outputs, 1)
                preds = torch.cat([preds, batch_preds]) if preds is not None else batch_preds

            return preds.to(device="cpu").numpy()


class FFNN(Neural):
    """
    FFNN with numpy inputs
    """

    steps: OrderedDict

    binary: bool

    def __init__(self, **config) -> None:

        super().__init__(**config)

        self.steps = OrderedDict(config['steps'])
        self.model = nn.Sequential(self.steps)
        self.model.to(self.device)

        self.loss = nn.BCEWithLogitsLoss() if config['binary'] else nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)


class LSTM(Neural):
    """
    LSTM with numpy inputs
    """

    input_size: int
    hidden_size: int
    num_layers: int

    num_classes: int

    def __init__(self, **config) -> None:

            super().__init__(**config)

            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            self.input_size = config['input_size']
            self.hidden_size = config['hidden_size']

            self.num_layers = config['num_layers']
            self.num_classes = config['num_classes']

            self.model = _LSTM(self.input_size, self.hidden_size, 
                               self.num_layers, self.num_classes, self.device)
            self.model.to(self.device)

            self.loss = nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)


class _LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes, device) -> None:
        
        super(_LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.device = device

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.relu(out)
        out = out[:, -1, :]
        return self.linear(out)
