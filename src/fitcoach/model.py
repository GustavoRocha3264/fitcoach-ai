"""Small LSTM classifier over per-frame landmark sequences.

Input  : (B, T, 33, 4) — windows of landmarks
Output : (B, num_classes) — class logits

For the Day 5 dataset (459 windows, 3 classes) we don't want anything big.
A single-layer LSTM with hidden_size=32 yields ~22 k trainable params,
which is comfortably below the parameter-count guard in the tests.
"""
from __future__ import annotations

import torch
from torch import nn


NUM_LANDMARKS = 33
NUM_CHANNELS = 4   # x, y, z, visibility
INPUT_DIM = NUM_LANDMARKS * NUM_CHANNELS  # 132


class LandmarkLSTM(nn.Module):
    """Sequence model: flatten landmarks per frame, run an LSTM, classify."""

    def __init__(
        self,
        *,
        hidden_size: int = 32,
        num_layers: int = 1,
        num_classes: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=INPUT_DIM,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 33, 4) → (B, T, 132)
        b, t = x.shape[0], x.shape[1]
        x = x.reshape(b, t, INPUT_DIM)
        _, (h_n, _) = self.lstm(x)
        # Use the last layer's final hidden state.
        last = h_n[-1]  # (B, hidden_size)
        return self.head(last)
