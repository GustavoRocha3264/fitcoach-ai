"""Tests for the training step / loop."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fitcoach.model import LandmarkLSTM  # noqa: E402
from fitcoach.train import train_step, evaluate  # noqa: E402


def test_train_step_returns_loss_and_reduces_it() -> None:
    """Repeated train_step calls on the same batch must drive the loss down."""
    torch.manual_seed(0)
    model = LandmarkLSTM(hidden_size=16, num_classes=3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.randn(8, 30, 33, 4)
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])

    first = train_step(model, x, y, opt)
    assert isinstance(first, float) and first > 0
    for _ in range(50):
        last = train_step(model, x, y, opt)
    assert last < first / 2, f"loss didn't decrease meaningfully: {first} → {last}"


def test_train_step_updates_parameters() -> None:
    torch.manual_seed(0)
    model = LandmarkLSTM(hidden_size=16, num_classes=3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    x = torch.randn(4, 30, 33, 4)
    y = torch.tensor([0, 1, 2, 0])
    train_step(model, x, y, opt)
    after = model.state_dict()
    # At least one parameter must have moved.
    diffs = [(after[k] - before[k]).abs().sum().item() for k in before]
    assert max(diffs) > 0


def test_evaluate_returns_loss_and_accuracy() -> None:
    torch.manual_seed(0)
    model = LandmarkLSTM(hidden_size=8, num_classes=3)
    # Build a tiny DataLoader-equivalent: list of (x, y) batches.
    batches = [(torch.randn(4, 30, 33, 4), torch.tensor([0, 1, 2, 0])) for _ in range(3)]
    loss, acc = evaluate(model, batches)
    assert isinstance(loss, float)
    assert 0.0 <= acc <= 1.0


def test_evaluate_perfect_on_overfit_batch() -> None:
    """After overfitting one batch, evaluate on that batch must report acc=1.0."""
    torch.manual_seed(0)
    model = LandmarkLSTM(hidden_size=32, num_classes=3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.randn(6, 30, 33, 4)
    y = torch.tensor([0, 1, 2, 0, 1, 2])
    for _ in range(300):
        train_step(model, x, y, opt)
    _, acc = evaluate(model, [(x, y)])
    assert acc == 1.0


def test_evaluate_does_not_train_the_model() -> None:
    """Calling evaluate must not change model weights (must run with no_grad)."""
    torch.manual_seed(0)
    model = LandmarkLSTM(hidden_size=8, num_classes=3)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    batches = [(torch.randn(2, 30, 33, 4), torch.tensor([0, 1]))]
    evaluate(model, batches)
    after = model.state_dict()
    for k in before:
        torch.testing.assert_close(before[k], after[k])
