"""Compare the PyTorch and Keras landmark-LSTM implementations.

Loads the trained PyTorch checkpoint, builds the equivalent (untrained) Keras
model, and reports:
- parameter counts
- output shape parity
- agreement on argmax over a fixed random batch (architectural equivalence
  rather than weight equivalence — gate ordering differs)
- inference latency over many runs

Usage:
    python scripts/compare_models.py
    python scripts/compare_models.py --batch 32 --runs 50
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare PyTorch vs Keras LSTM inference")
    p.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "classifier.pt")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--window-size", type=int, default=30)
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _time_torch(model, x, runs: int) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(3):  # warm up
            model(x)
        t0 = time.perf_counter()
        for _ in range(runs):
            model(x)
        return (time.perf_counter() - t0) / runs


def _time_keras(model, x_np, runs: int) -> float:
    for _ in range(3):  # warm up
        model.predict(x_np, verbose=0)
    t0 = time.perf_counter()
    for _ in range(runs):
        model.predict(x_np, verbose=0)
    return (time.perf_counter() - t0) / runs


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from fitcoach.model import LandmarkLSTM
    from fitcoach.model_tf import build_landmark_lstm

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    hidden = cfg["hidden_size"]
    n_classes = cfg["num_classes"]
    print(f"Loaded checkpoint: hidden={hidden} classes={n_classes} "
          f"val_acc@best={ckpt.get('val_acc'):.2%}")

    torch_model = LandmarkLSTM(hidden_size=hidden, num_classes=n_classes)
    torch_model.load_state_dict(ckpt["model_state"])
    keras_model = build_landmark_lstm(hidden_size=hidden, num_classes=n_classes)

    n_torch = sum(p.numel() for p in torch_model.parameters() if p.requires_grad)
    n_keras = int(np.sum([np.prod(w.shape) for w in keras_model.trainable_weights]))
    print(f"\nParameters: torch={n_torch:,}  keras={n_keras:,}")

    x_np = np.random.randn(args.batch, args.window_size, 33, 4).astype(np.float32)
    x_pt = torch.from_numpy(x_np)

    with torch.no_grad():
        out_torch = torch_model(x_pt).numpy()
    out_keras = keras_model.predict(x_np, verbose=0)
    print(f"Output shape: torch={out_torch.shape}  keras={out_keras.shape}")

    # Architectural equivalence: same shape, both produce class logits.
    # We can't expect numerical agreement (weights differ; Keras is untrained
    # here) — instead we sanity-check that the Keras model produces a valid
    # argmax over each row.
    keras_argmax = out_keras.argmax(axis=-1)
    assert keras_argmax.shape == (args.batch,) and keras_argmax.max() < n_classes

    print("\nLatency (per batch, lower is better):")
    pt_ms = _time_torch(torch_model, x_pt, args.runs) * 1000
    kr_ms = _time_keras(keras_model, x_np, args.runs) * 1000
    print(f"  torch (cpu)  : {pt_ms:7.2f} ms / batch")
    print(f"  keras (cpu)  : {kr_ms:7.2f} ms / batch")

    # MPS run for torch, optional but nice to have on the demo machine.
    if torch.backends.mps.is_available():
        m = torch_model.to("mps")
        x_mps = x_pt.to("mps")
        pt_mps_ms = _time_torch(m, x_mps, args.runs) * 1000
        print(f"  torch (mps)  : {pt_mps_ms:7.2f} ms / batch")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
