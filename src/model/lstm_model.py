import torch
import random
import torch.nn as nn


class HydroForecast(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=48, hidden_size=12, num_layers=5, batch_first=True)
        self.linear = nn.Linear(12, 7)

    def forward(self, x):
        x, _ = self.lstm(x)
        # extract only the last time step
        x = x[:, -1, :]
        x = self.linear(x)
        return x
