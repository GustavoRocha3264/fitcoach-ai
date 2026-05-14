"""Train the LandmarkLSTM classifier on the extracted landmark windows.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --epochs 30 --hidden 64 --lr 1e-3
    python scripts/train_model.py --val-suffix _4 --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fitcoach.dataset import (  # noqa: E402
    EXERCISE_LABELS,
    LandmarkDataset,
    scan_landmarks_dir,
)
from fitcoach.model import LandmarkLSTM  # noqa: E402
from fitcoach.split import split_entries_by_suffix  # noqa: E402
from fitcoach.train import evaluate, train_step  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the exercise classifier")
    p.add_argument("--landmarks", type=Path, default=ROOT / "data" / "processed" / "landmarks")
    p.add_argument("--out", type=Path, default=ROOT / "models" / "classifier.pt")
    p.add_argument("--val-suffix", default="_4",
                   help="hold out the clip ending with this suffix per class")
    p.add_argument("--window-size", type=int, default=30)
    p.add_argument("--stride", type=int, default=15)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-coverage", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    entries = scan_landmarks_dir(args.landmarks, min_coverage=args.min_coverage)
    print(f"Loaded {len(entries)} clips (min_coverage={args.min_coverage})")
    train_entries, val_entries = split_entries_by_suffix(entries, val_suffix=args.val_suffix)
    print(f"  train: {len(train_entries)} clips — "
          f"{[e.path.stem for e in train_entries]}")
    print(f"  val  : {len(val_entries)} clips — "
          f"{[e.path.stem for e in val_entries]}")

    train_ds = LandmarkDataset(train_entries, window_size=args.window_size, stride=args.stride)
    val_ds   = LandmarkDataset(val_entries,   window_size=args.window_size, stride=args.stride)
    print(f"  train windows: {len(train_ds)}  val windows: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    model = LandmarkLSTM(
        hidden_size=args.hidden,
        num_classes=len(EXERCISE_LABELS),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: hidden={args.hidden}  trainable params: {n_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    history: list[dict] = []
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        n_batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            running += train_step(model, x, y, opt)
            n_batches += 1
        train_loss = running / max(n_batches, 1)

        val_loss, val_acc = evaluate(
            model, ((x.to(device), y.to(device)) for x, y in val_loader)
        )
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "val_acc": val_acc})
        marker = "*" if val_acc > best_val_acc else " "
        print(f"  epoch {epoch:3d}  train_loss={train_loss:.3f}  "
              f"val_loss={val_loss:.3f}  val_acc={val_acc:.2%} {marker}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "config": {
                    "hidden_size": args.hidden,
                    "num_classes": len(EXERCISE_LABELS),
                    "window_size": args.window_size,
                },
                "val_acc": val_acc,
                "epoch": epoch,
                "exercise_labels": EXERCISE_LABELS,
            }, args.out)

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s. Best val_acc: {best_val_acc:.2%}.")
    print(f"Checkpoint: {args.out}")
    (args.out.with_suffix(".history.json")).write_text(json.dumps(history, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
