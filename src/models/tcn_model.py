"""Simple Temporal Convolutional Network implementation."""
from __future__ import annotations

import torch
from torch import nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):  # type: ignore[override]
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):  # type: ignore[override]
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs: int, channels: list[int], kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        num_levels = len(channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else channels[i - 1]
            out_channels = channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, dilation=dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):  # type: ignore[override]
        return self.network(x)


class TCNClassifier(nn.Module):
    def __init__(self, num_inputs: int, channels: list[int], dropout: float = 0.2):
        super().__init__()
        self.tcn = TemporalConvNet(num_inputs, channels, dropout=dropout)
        self.classifier = nn.Linear(channels[-1], 1)

    def forward(self, x):  # type: ignore[override]
        features = self.tcn(x)
        last = features[:, :, -1]
        logits = self.classifier(last)
        return logits.squeeze(-1)
