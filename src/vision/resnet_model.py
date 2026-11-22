"""
Unit IV: ResNet Architecture for Sign Language Gesture Classification.
Residual blocks with identity shortcut connections, batch normalization, and dropout.
Aligned to CSAI3027P Deep Learning - Project 13: Sign Language Assistant.
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    """
    1D Residual Block for Spatio-Temporal Gesture Sequences:
    y = F(x) + shortcut(x)
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class SignResNet(nn.Module):
    """
    Deep Residual Network tailored for Sign Language temporal landmark sequences:
    Input: (Batch, Time=25, Features=126) -> permuted to (Batch, 126, 25)
    Stages:
      - Stem Conv: 126 -> 64
      - Stage 1: ResBlock(64 -> 64)
      - Stage 2: ResBlock(64 -> 128, stride=2)
      - Stage 3: ResBlock(128 -> 256, stride=2)
      - Global Adaptive Average Pooling
      - Classification Head: Linear(256 -> n_classes)
    """

    def __init__(self, in_features: int = 126, num_classes: int = 98, dropout: float = 0.3):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_features, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        # Residual stages
        self.layer1 = ResidualBlock1D(64, 64, stride=1)
        self.layer2 = ResidualBlock1D(64, 128, stride=2)
        self.layer3 = ResidualBlock1D(128, 256, stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, 25, 126) -> permute to (Batch, 126, 25)
        if x.dim() == 3 and x.shape[1] == 25 and x.shape[2] == self.in_features:
            x = x.permute(0, 2, 1)

        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.pool(out).squeeze(-1)
        out = self.drop(out)
        logits = self.classifier(out)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
