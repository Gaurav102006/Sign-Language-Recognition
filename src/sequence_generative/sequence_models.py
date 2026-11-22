"""
Unit V: Sequence Models for Sign Language Temporal Translation.
Deep RNN, Multi-layer BiLSTM, and Transformer Sequence Decoder.
Includes Step vs Sigmoid vs Tanh gating mechanism ablation.
Aligned to CSAI3027P Deep Learning - Project 13: Sign Language Assistant.
"""

from typing import Dict, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepRNNSignClassifier(nn.Module):
    """
    Multi-layer Recurrent Neural Network with selectable non-linearity:
    tanh or relu or custom activation gating.
    """

    def __init__(
        self,
        in_dim: int = 126,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 98,
        nonlinearity: str = "tanh",
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.RNN(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Seq_len=25, Dim=126)
        out, _ = self.rnn(x)
        last_step = out[:, -1, :]  # Take final temporal state
        return self.fc(last_step)


class BiLSTMSignClassifier(nn.Module):
    """
    Bidirectional Long Short-Term Memory (BiLSTM) Network.
    Gating mechanisms: input gate (i), forget gate (f), output gate (o), candidate cell (c).
    Explicitly utilizes sigmoid for gates and tanh for memory candidate states.
    """

    def __init__(
        self,
        in_dim: int = 126,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 98,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (h_n, c_n) = self.lstm(x)
        # Concatenate forward and backward final hidden states
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        h_cat = torch.cat([h_forward, h_backward], dim=-1)
        h_cat = self.dropout(h_cat)
        return self.fc(h_cat)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TransformerSignDecoder(nn.Module):
    """
    Transformer Sequence Encoder/Decoder for Sign Language translation.
    Multi-Head Self-Attention (MHSA) capturing long-range temporal joint correlations.
    """

    def __init__(
        self,
        in_dim: int = 126,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        num_classes: int = 98,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=50)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Seq_len=25, Dim=126)
        x_proj = self.input_proj(x)
        x_pos = self.pos_encoder(x_proj)
        encoded = self.transformer(x_pos)
        pooled = torch.mean(encoded, dim=1)  # Mean pooling over time
        return self.fc(pooled)


class GatingAblationExperiment:
    """
    Ablation analysis comparing Step, Sigmoid, and Tanh gating behaviors in recurrent cells:
    - Step: Non-differentiable (zero gradient), fails backpropagation through time.
    - Sigmoid: Non-negative [0, 1] gating, ideal for memory gates (forget/input).
    - Tanh: Zero-centered [-1, +1], maintains zero-centered activations and prevents bias drift.
    """

    @staticmethod
    def evaluate_cell_dynamics(n_steps: int = 25) -> Dict[str, list]:
        """Simulates recurrent hidden state magnitude over 25 steps under different activations."""
        w_rec = 1.1  # slightly unstable weight
        h_init = 0.5

        history = {"step": [], "sigmoid": [], "tanh": []}

        # Sigmoid recurrence
        h = h_init
        for _ in range(n_steps):
            h = 1.0 / (1.0 + math.exp(-w_rec * h))
            history["sigmoid"].append(round(h, 4))

        # Tanh recurrence
        h = h_init
        for _ in range(n_steps):
            h = math.tanh(w_rec * h)
            history["tanh"].append(round(h, 4))

        # Step recurrence
        h = h_init
        for _ in range(n_steps):
            h = 1.0 if (w_rec * h) >= 0.5 else 0.0
            history["step"].append(h)

        return history
