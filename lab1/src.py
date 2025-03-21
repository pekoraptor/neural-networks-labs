import torch.utils.data as data
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np


def prepare_data(batch_size, train_path, test_X_path, test_y_path, delimiter=","):
    train = pd.read_csv(train_path, delimiter=delimiter)
    test_X = pd.read_csv(test_X_path, delimiter=delimiter)
    test_y = pd.read_csv(test_y_path, header=None)
    train_dataset = data.TensorDataset(
        torch.from_numpy(train.values[:, 2:-3].astype(float)),
        torch.from_numpy(train.values[:, -1].astype(float)),
    )
    test_dataset = data.TensorDataset(
        torch.from_numpy(test_X.values[:, 1:].astype(float)),
        torch.from_numpy(test_y.values.astype(float)),
    )
    return (
        data.DataLoader(
            train_dataset, batch_size=batch_size, drop_last=True, shuffle=True
        ),
        test_dataset,
    )


class Perceptron(nn.Module):
    def __init__(
        self,
        num_inputs=12,
        num_hidden1=64,
        num_hidden2=32,
        num_outputs=1,
    ):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, num_hidden1)
        self.act_fn1 = nn.Tanh()
        self.linear2 = nn.Linear(num_hidden1, num_hidden2)
        self.act_fn2 = nn.Tanh()
        self.linear3 = nn.Linear(num_hidden2, num_outputs)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act_fn1(x)
        x = self.linear2(x)
        x = self.act_fn2(x)
        x = self.linear3(x)
        return x


def rmsle(y_true, y_pred):
    n = len(y_true)
    msle = np.mean(
        [
            (np.log(max(y_pred[i], 0) + 1) - np.log(y_true[i] + 1)) ** 2.0
            for i in range(n)
        ]
    )
    return np.sqrt(msle)
