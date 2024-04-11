# Simply neural models for the text experiments
# Alejandro Ciuba, alejandrociuba@pitt.edu
from collections import OrderedDict
from torch.utils.data import (DataLoader,
                              Dataset, )

import torch

import pandas as pd
import torch.nn as nn


class IterableDataset(Dataset):

    X: torch.Tensor
    y: torch.Tensor

    def __init__(self, X, y):

        super().__init__()

        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index, :], self.y[index]


class FFNN:

    model: nn.Module
    steps: OrderedDict
    num_classes: int
    batch_size: int

    device: str

    crit = None
    opt = None

    def __init__(self, input_size: int, hidden_sizes: list[int], num_classes: int, lr: float) -> None:

        self.num_classes = num_classes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        steps = [("l1", nn.Linear(input_size, hidden_sizes[0])),
                 ("relu1", nn.ReLU()), ]

        for i, size in enumerate(hidden_sizes):

            if i == len(hidden_sizes) - 1:
                continue

            steps.append((f"l{i+2}", nn.Linear(size, hidden_sizes[i + 1])))
            steps.append((f"relu{i+2}", nn.ReLU()))
        
        steps.append((f"l{len(hidden_sizes)+1}", nn.Linear(hidden_sizes[-1], num_classes)))

        self.steps = OrderedDict(steps)
        self.model = nn.Sequential(self.steps)
        self.model.to(self.device)

        self.crit = nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam(params=self.model.parameters(), lr=lr)

    def fit(self, X, y, epochs: int, batch_size: int):

        self.batch_size = batch_size
        
        dataset = IterableDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

        n_total_steps = len(dataloader)

        for epoch in range(epochs):
            
            for i, (X, y) in enumerate(dataloader):

                X = X.to(self.device)
                y = y.to(self.device)

                outputs = self.model(X)
                loss = self.crit(outputs, y)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                if (i + 1) % 100 == 0:
                    print('epoch: %d/%d, step: %d/%d, loss=%.4ff' % (epoch+1, epochs, i+1, n_total_steps, loss.item()))

    def predict(self, X) -> tuple[list, list]:

        dataset = IterableDataset(X, torch.zeros(X.shape[0]))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

        with torch.no_grad():

            for X, y in dataloader:

                X = X.to(self.device)
                y = y.to(self.device)
                outputs = self.model(X)

                _, preds = torch.max(outputs, 1)

            return list(preds)
