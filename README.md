# FitCoach AI

A small end-to-end computer-vision project that turns a video of someone
exercising into a structured, scored workout session: detect the body, count
the reps, score the form, classify the exercise, and persist the results.

```
        ┌───────────┐   ┌──────────────┐   ┌─────────────┐   ┌─────────────┐
 video ─► MediaPipe ├──►│ joint angle  ├──►│ rep counter ├──►│   form      │
        │  pose +   │   │ + landmark   │   │   (FSM)     │   │ evaluator   │
        │  segment  │   │   windows    │   └──────┬──────┘   └──────┬──────┘
        └─────┬─────┘   └──────┬───────┘          │                 │
              │                │                  │                 │
              ▼                ▼                  ▼                 ▼
        skeleton +        LSTM classifier   Postgres (sessions, reps)
        bg-blur HUD       (3 exercises)     Mongo (raw landmark blobs)
```

The codebase is built test-first (TDD): every module has a paired test file
written before the implementation.

---

## What it actually does

Given a video of a person doing **squats**, **push-ups**, or **biceps curls**:

1. **MediaPipe Pose** finds the 33-landmark BlazePose skeleton plus a body
   segmentation mask, frame by frame.
2. **Joint-angle math** (hip–knee–ankle for squats, shoulder–elbow–wrist for
   push-ups and curls) produces a 1-D angle stream.
3. **A finite-state machine with hysteresis** counts reps: drop below
   `down_threshold` arms a rep, return above `up_threshold` completes it.
   NaN samples (occluded frames) are ignored.
4. **A rule-based form evaluator** scores each rep on depth — pass/fail and a
   0–1 score scaled from "barely passing" to "ideal".
5. **A small PyTorch LSTM** (~21 k parameters) classifies the exercise from
   the 30-frame landmark window, trained on a curated set of 12 public clips.
6. **Persistence**: Postgres stores session metadata + per-rep records; Mongo
   stores the raw `(T, 33, 4)` landmark tensors per session.
7. **A C++ pybind11 module** mirrors the joint-angle math in native code with
   a ~2.4× speedup over NumPy on per-call invocation.

---

## Quick start

The project targets Apple Silicon (arm64) with no NVIDIA hardware; PyTorch
uses the **MPS (Metal)** backend and TensorFlow uses **tensorflow-metal**.

```bash
# 1) Create the conda environment (~10 min: PyTorch + TF + MediaPipe wheels).
brew install --cask miniforge          # if you don't have conda yet
conda env create -f environment.yml
conda activate fitcoach

# 2) Start Postgres + Mongo in Docker.
cp .env.example .env
docker compose up -d

# 3) Fetch the sample exercise videos.
#    Note: most clips have url:null and must be dropped into data/raw/videos/
#    manually — see data/videos.json for the source URLs and naming.
python scripts/fetch_videos.py

# 4) Extract per-frame landmark tensors into data/processed/landmarks/.
python scripts/extract_landmarks.py

# 5) Try the demos.
python scripts/demo_pose.py                       # skeleton + bg blur HUD
python scripts/demo_reps.py --exercise squat      # angle + rep count + form HUD

# 6) Train the classifier (4 s on MPS, 100% val acc on held-out clips).
python scripts/train_model.py

# 7) Compare PyTorch vs Keras inference.
python scripts/compare_models.py

# 8) Build the C++ extension and benchmark against NumPy.
bash scripts/build_cpp.sh
PYTHONPATH=cpp/build python scripts/bench_angle.py
```

A `KMP_DUPLICATE_LIB_OK=TRUE` environment variable is set in `.env` to
suppress a benign duplicate-libomp warning that comes from OpenCV / MediaPipe
sharing OpenMP with PyTorch on macOS.

---

## The pipeline, module by module

### `pose.py` — pose + segmentation
Wraps `mediapipe.tasks.vision.PoseLandmarker` (MediaPipe 0.10.35 removed the
legacy `mp.solutions` namespace) and exposes:
- `PoseDetector(model_variant=..., static_image_mode=...)`: VIDEO mode uses
  the in-clip tracker for higher coverage; IMAGE mode runs per-frame fresh.
- `model_variant ∈ {lite, full, heavy}` selects the trained model. **Full** is
  the default; in benchmarking, heavy was *more conservative* on
  cropped-frame clips and actually reduced coverage, so it isn't worth using
  by default for this dataset.

### `drawing.py` — overlay
Skeleton + connections, segmentation-mask background blur, FPS HUD.

### `angles.py` — joint angles
Pure NumPy. `joint_angle(a, b, c)` returns the angle at `b` in degrees;
`landmark_angle(a, b, c)` short-circuits to NaN when any landmark's
visibility is below threshold. `knee_angle` / `elbow_angle` are convenience
helpers using BlazePose indices.

### `rep_counter.py` — FSM
Two-state machine (up / down) with mandatory hysteresis
(`down_threshold < up_threshold`). Emits `RepEvent(rep_index, min_angle)`
on each completed rep. NaN samples skip the FSM untouched.

### `form.py` — rule-based evaluator
One `DepthRule` per exercise. A rep passes if its minimum angle is below the
exercise's `max_angle`; the score scales linearly from 0 at `max_angle` to 1
at `ideal_angle`. Easy to extend with more rules (back alignment, knee
tracking, etc.) once the landmark stream is trusted enough.

### `extract.py` — video → landmark tensor
Iterates frames, calls a detector, stacks to `(T, 33, 4)`. Undetected frames
contribute a NaN row, preserving temporal alignment for windowing later.

### `dataset.py` — windowed PyTorch dataset
- `EXERCISE_LABELS = {squat: 0, pushup: 1, curl: 2}` — part of the public
  contract; reordering would invalidate trained checkpoints.
- `LandmarkDataset` slides a `window_size`-frame window over each clip with
  configurable `stride`; NaN rows are replaced with `nan_fill` on the way
  out so the network never sees NaN.
- `exercise_from_stem` maps `squat`, `squat_2`, `squat_alt` → `squat` so an
  exercise can accumulate many clips.
- `clip_coverage(arr)` + `scan_landmarks_dir(..., min_coverage=...)` drops
  low-quality clips before they reach training.

### `split.py` — clip-level train/val
Splitting at the window level would leak (windows from the same clip share
statistics). `split_entries_by_suffix(entries, val_suffix="_4")` holds out
the `_4` clip per class and raises if any class is missing one. The seeded
`split_entries_by_count` variant exists for cross-validation later.

### `model.py` — PyTorch LSTM
Per-frame flatten (132 features) → single LSTM layer (hidden=32) → linear
head (3 classes). 21 347 trainable parameters; small on purpose for the
459-window dataset.

### `model_tf.py` — Keras counterpart
Same architecture in Keras for the framework-comparison demo
(`scripts/compare_models.py`). Not bit-identical to the PyTorch model:
LSTM gate ordering and bias parameterisation differ by 128 elements.

### `train.py` — training step + evaluation
- `train_step(model, x, y, optimizer)` runs one forward / backward step.
- `evaluate(model, batches)` reports average loss + accuracy under
  `no_grad`, never updating weights.

### `db/postgres.py` — `SessionStore`
Two tables (`sessions`, `reps`) with `ON DELETE CASCADE` and a
`UNIQUE (session_id, rep_index)` constraint so duplicate reps fail fast.
Doesn't own the connection — transaction boundaries stay with the caller.

### `db/mongo.py` — `LandmarkStore`
One document per session_id with `shape`, `dtype`, and the raw float32
bytes. `replace_one(upsert=True)` makes save idempotent. Load preserves NaN
rows.

### `cpp/angles.cpp` — native joint angle
pybind11 port of the NumPy `joint_angle`. Same NaN semantics, same acos
clamping for numerical safety. `scripts/build_cpp.sh` builds it with CMake +
ninja into `cpp/build/`. Benchmarked at ~2.4× faster than NumPy per call;
the gain is bounded by Python ↔ C++ marshaling — a vectorised batch API in
C++ would close the gap by an order of magnitude.

---

## Results so far

**Classification on held-out clips (one per class):**

| | Train | Val |
|---|---|---|
| Clips | 9 | 3 (`squat_4`, `pushup_4`, `curl_4`) |
| Windows | 399 | 60 |
| Accuracy | — | **100%** (epoch 11) |

`squat_4` / `pushup_4` / `curl_4` are entire clips the model never saw during
training, so this isn't window-level leakage. The val set is small (60
windows, 3 distinct motion patterns), so "100%" is a real number but
generalisation to unseen subjects and angles is unproven — that's
day-7-after work.

**Framework inference latency** (16-sample batch):

| Implementation | Time / batch |
|---|---|
| PyTorch CPU | 0.38 ms |
| PyTorch MPS | 0.23 ms |
| Keras CPU (eager) | 18.08 ms |

**C++ vs NumPy** (50 000 angle computations):

| Implementation | µs / call | Speedup |
|---|---|---|
| Python + NumPy | 2.25 | 1.00× |
| C++ via pybind11 | 0.94 | **2.40×** |

---

## Testing

The project follows test-first development. **95 tests total**, split into
two groups because TensorFlow and PyTorch fight over native libraries when
loaded into the same Python process:

```bash
# Everything except the TF parity tests.
pytest tests/ --ignore=tests/test_model_tf.py

# TF parity tests in their own process.
pytest tests/test_model_tf.py
```

Tests are designed to skip cleanly when their dependencies aren't ready:

- `tests/test_db_postgres.py` and `test_db_mongo.py` skip if the docker
  containers aren't reachable.
- `tests/test_angles_cpp.py` skips if `cpp/build/` doesn't contain the
  compiled extension.

So a fresh clone with no Docker and no C++ toolchain still gets a green
suite, just with fewer tests collected.

---

## Repository layout

```
fitcoach-ai/
├── cpp/                # C++ pybind11 sources + CMakeLists.txt
├── data/
│   ├── videos.json     # manifest of sample clips (source URLs, attribution)
│   ├── raw/videos/     # mp4 files (gitignored)
│   └── processed/landmarks/  # (T, 33, 4) .npy files (gitignored)
├── models/             # MediaPipe .task + trained classifier.pt (gitignored)
├── notebooks/          # exploration (not imported by src/)
├── scripts/            # CLIs: fetch, extract, train, compare, build_cpp, …
├── src/fitcoach/       # The Python package itself
│   ├── pose.py
│   ├── drawing.py
│   ├── angles.py
│   ├── rep_counter.py
│   ├── form.py
│   ├── extract.py
│   ├── dataset.py
│   ├── split.py
│   ├── model.py
│   ├── model_tf.py
│   ├── train.py
│   ├── db/
│   │   ├── postgres.py
│   │   └── mongo.py
│   └── classifier/
├── tests/              # pytest suite mirroring src/fitcoach/
├── docker-compose.yml  # Postgres 16 + Mongo 7
├── environment.yml     # conda env definition
└── .env.example
```

---

## Caveats, known limitations, and obvious follow-ups

- **Dataset is small** (12 public clips, ~459 windows). The 100% val accuracy
  is honest but the held-out set is tiny; a larger split across more subjects
  and camera angles would yield a more trustworthy number.
- **Form evaluator scores depth only.** Real coaching needs back-alignment,
  knee-tracking, foot placement, etc. The depth threshold per exercise is a
  reasonable starting point; the framework is structured to make adding more
  `Rule` types straightforward.
- **The C++ angle module isn't wired into the live demo yet.** The Python
  `fitcoach.angles.joint_angle` still calls NumPy. Switching the import to
  prefer the C++ path when built is a one-line follow-up.
- **The Keras model is untrained** in `compare_models.py`. The comparison is
  architectural and latency-based, not numerical. Training both with the
  same recipe and recording per-epoch losses would close that loop.
- **Webcam path exists but is unmaintained.** Demos default to video files
  for determinism; `--source 0` still works but is not covered by tests.

---

## Built with

- Python 3.11 in a Miniforge conda env
- MediaPipe 0.10 (Tasks API, BlazePose 33-landmark model)
- OpenCV 4.13
- PyTorch 2.12 (MPS backend), TensorFlow 2.18 + tensorflow-metal 1.2
- Postgres 16 and MongoDB 7 via Docker Compose
- C++20 + pybind11 + CMake/ninja
- pytest, ruff, mypy
