"""Keras counterpart of `LandmarkLSTM` for the framework comparison.

Mirrors the PyTorch architecture: per-frame flatten → LSTM → Linear head.
The two implementations are architecturally equivalent but not bit-identical
(PyTorch and TF differ on LSTM gate ordering and bias parameterisation),
so we compare them at the level of param count, output shape, predicted
class on the same input, and inference latency.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


NUM_LANDMARKS = 33
NUM_CHANNELS = 4
INPUT_DIM = NUM_LANDMARKS * NUM_CHANNELS  # 132


def build_landmark_lstm(
    *,
    hidden_size: int = 32,
    num_classes: int = 3,
    num_layers: int = 1,
    dropout: float = 0.0,
) -> keras.Model:
    """Build the Keras model. Input: (B, T, 33, 4). Output: (B, num_classes)."""
    inputs = keras.Input(shape=(None, NUM_LANDMARKS, NUM_CHANNELS))
    x = layers.Reshape((-1, INPUT_DIM))(inputs)
    for i in range(num_layers):
        is_last = i == num_layers - 1
        x = layers.LSTM(
            hidden_size,
            return_sequences=not is_last,
            dropout=dropout,
        )(x)
    outputs = layers.Dense(num_classes)(x)
    return keras.Model(inputs, outputs, name="landmark_lstm_tf")
