"""Tests for the Keras LSTM that mirrors the PyTorch model."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fitcoach.model_tf import build_landmark_lstm  # noqa: E402


def test_keras_forward_shape() -> None:
    model = build_landmark_lstm(hidden_size=32, num_classes=3)
    x = np.random.randn(8, 30, 33, 4).astype(np.float32)
    out = model.predict(x, verbose=0)
    assert out.shape == (8, 3)
    assert out.dtype == np.float32


def test_keras_accepts_arbitrary_sequence_length() -> None:
    """T should be free; only the per-frame feature size is fixed."""
    model = build_landmark_lstm(hidden_size=16, num_classes=3)
    short = model.predict(np.random.randn(2, 10, 33, 4).astype(np.float32), verbose=0)
    long  = model.predict(np.random.randn(2, 90, 33, 4).astype(np.float32), verbose=0)
    assert short.shape == long.shape == (2, 3)


def test_keras_param_count_is_close_to_pytorch_equivalent() -> None:
    """Same hidden size on the same architecture should have roughly the same
    number of trainable params in either framework (off by a small constant
    is fine; differences come from how bias is parameterised)."""
    import torch

    from fitcoach.model import LandmarkLSTM

    keras_model = build_landmark_lstm(hidden_size=32, num_classes=3)
    torch_model = LandmarkLSTM(hidden_size=32, num_classes=3)
    n_keras = int(np.sum([np.prod(w.shape) for w in keras_model.trainable_weights]))
    n_torch = sum(p.numel() for p in torch_model.parameters() if p.requires_grad)
    # Allow a 30% gap to absorb framework-specific bias layouts.
    assert abs(n_keras - n_torch) / max(n_keras, n_torch) < 0.30, \
        f"param counts diverge: keras={n_keras}, torch={n_torch}"
