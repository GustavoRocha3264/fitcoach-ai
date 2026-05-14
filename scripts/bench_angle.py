"""Benchmark NumPy vs C++ joint_angle implementations.

Times both on N random triplets, reports per-call time and speedup.

Usage:
    PYTHONPATH=cpp/build python scripts/bench_angle.py
    PYTHONPATH=cpp/build python scripts/bench_angle.py --n 100000 --warmup 1000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build"))

from fitcoach.angles import joint_angle as joint_angle_py  # noqa: E402

try:
    import fitcoach_cpp  # type: ignore[import-not-found]
except ImportError:
    raise SystemExit(
        "fitcoach_cpp not found. Build it first with: bash scripts/build_cpp.sh"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark joint_angle: NumPy vs C++")
    p.add_argument("--n", type=int, default=50_000, help="number of triplets")
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    triplets = rng.standard_normal((args.n, 3, 2)).astype(np.float64)

    # Warm up both code paths.
    for i in range(args.warmup):
        a, b, c = triplets[i % args.n]
        joint_angle_py(a, b, c)
        fitcoach_cpp.joint_angle(tuple(a), tuple(b), tuple(c))

    # NumPy
    t0 = time.perf_counter()
    for a, b, c in triplets:
        joint_angle_py(a, b, c)
    t_py = time.perf_counter() - t0

    # C++
    t0 = time.perf_counter()
    for a, b, c in triplets:
        fitcoach_cpp.joint_angle(tuple(a), tuple(b), tuple(c))
    t_cpp = time.perf_counter() - t0

    print(f"n = {args.n:,} angle computations")
    print(f"  python+numpy : {t_py*1000:8.1f} ms total  ({t_py/args.n*1e6:6.2f} µs/call)")
    print(f"  c++          : {t_cpp*1000:8.1f} ms total  ({t_cpp/args.n*1e6:6.2f} µs/call)")
    print(f"  speedup      : {t_py / t_cpp:6.2f}×")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
