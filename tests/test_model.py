"""Tests for the PyTorch LSTM classifier."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fitcoach.model import LandmarkLSTM, NUM_LANDMARKS, NUM_CHANNELS  # noqa: E402


def test_constants_match_pipeline() -> None:
    """The model expects the same landmark shape the extractor produces."""
    assert NUM_LANDMARKS == 33
    assert NUM_CHANNELS == 4


def test_forward_shape() -> None:
    model = LandmarkLSTM(hidden_size=32, num_classes=3)
    x = torch.randn(8, 30, 33, 4)
    logits = model(x)
    assert logits.shape == (8, 3)
    assert logits.dtype == torch.float32


def test_forward_accepts_arbitrary_sequence_length() -> None:
    """The LSTM should work on any T, not just T=30."""
    model = LandmarkLSTM(hidden_size=16, num_classes=3)
    short = model(torch.randn(2, 10, 33, 4))
    long = model(torch.randn(2, 90, 33, 4))
    assert short.shape == long.shape == (2, 3)


def test_parameter_count_is_small() -> None:
    """Sanity: a 'small LSTM' should be under ~50k params for our dataset size."""
    model = LandmarkLSTM(hidden_size=32, num_classes=3)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n < 50_000, f"model too large for the dataset: {n} params"


def test_backward_updates_parameters() -> None:
    """One backward step must produce non-zero gradients on every parameter."""
    model = LandmarkLSTM(hidden_size=16, num_classes=3)
    x = torch.randn(4, 30, 33, 4)
    y = torch.tensor([0, 1, 2, 0])
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_deterministic_under_seed() -> None:
    """Same seed + same input ⇒ identical logits."""
    torch.manual_seed(0)
    m1 = LandmarkLSTM(hidden_size=16, num_classes=3)
    torch.manual_seed(0)
    m2 = LandmarkLSTM(hidden_size=16, num_classes=3)
    x = torch.randn(3, 30, 33, 4)
    torch.testing.assert_close(m1(x), m2(x))
