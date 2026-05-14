"""Training step and evaluation loop for the LSTM classifier."""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def train_step(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Run one forward/backward/optimizer step. Returns the (scalar) loss."""
    model.train()
    optimizer.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def evaluate(
    model: nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[float, float]:
    """Compute average loss and accuracy over the given (x, y) batches.

    `batches` can be a DataLoader or any iterable of (x, y) tuples.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in batches:
            logits = model(x)
            total_loss += float(F.cross_entropy(logits, y, reduction="sum").item())
            preds = logits.argmax(dim=-1)
            correct += int((preds == y).sum().item())
            total += int(y.shape[0])
    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total
