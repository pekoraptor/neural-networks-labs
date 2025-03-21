import torch.utils.data as data
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np


def data_processing():
    df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
        delimiter=";",
    )
    train = df.sample(frac=0.8, random_state=200)  # random state is a seed value
    test = df.drop(train.index)
    train_dataset = data.TensorDataset(
        torch.from_numpy(train.values[:, :-1]), torch.from_numpy(train.values[:, -1])
    )
    test_dataset = data.TensorDataset(
        torch.from_numpy(test.values[:, :-1]), torch.from_numpy(test.values[:, -1])
    )
    return train_dataset, test_dataset


class Perceptron(nn.Module):
    def __init__(
        self,
        num_inputs=11,
        num_hidden1=256,
        num_hidden2=128,
        num_hidden3=32,
        num_outputs=11,
    ):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, num_hidden1)
        self.act_fn1 = nn.Tanh()
        self.linear2 = nn.Linear(num_hidden1, num_hidden2)
        self.act_fn2 = nn.Tanh()
        self.linear3 = nn.Linear(num_hidden2, num_hidden3)
        self.act_fn3 = nn.Tanh()
        self.linear4 = nn.Linear(num_hidden3, num_outputs)
        # self.output_act_fn = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act_fn1(x)
        x = self.linear2(x)
        x = self.act_fn2(x)
        x = self.linear3(x)
        x = self.act_fn3(x)
        x = self.linear4(x)
        # x = self.output_act_fn(x)
        return x


if __name__ == "__main__":
    model = Perceptron()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    loss_module = nn.CrossEntropyLoss()

    train_dataset, test_dataset = data_processing()
    train_loader = data.DataLoader(
        train_dataset, batch_size=128, drop_last=True, shuffle=True
    )
    test_loader = data.DataLoader(
        test_dataset, batch_size=128, drop_last=True, shuffle=True
    )

    model.train()
    for epoch in range(1200):
        for data_inputs, data_labels in train_loader:
            preds = model(data_inputs.float())
            # preds = preds.squeeze(dim=1)
            loss = loss_module(preds, data_labels.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch}, loss: {loss.item():.3}")
    print(data_labels)
    print(preds)

    model.eval()
    true_preds, num_preds = 0.0, 0.0

    with torch.no_grad():
        for data_inputs, data_labels in test_loader:
            preds = model(data_inputs.float())
            preds = preds.squeeze(dim=1)
            true_preds += torch.sum(torch.argmax(preds, dim=1) == data_labels)
            num_preds += len(data_labels)
    print(f"Accuracy (test): {true_preds/num_preds:.3}")

    with torch.no_grad():
        for data_inputs, data_labels in train_loader:
            preds = model(data_inputs.float())
            preds = preds.squeeze(dim=1)
            true_preds += torch.sum(torch.argmax(preds, dim=1) == data_labels)
            num_preds += len(data_labels)
    print(f"Accuracy (train): {true_preds/num_preds:.3}")
