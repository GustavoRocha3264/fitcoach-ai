#!/usr/bin/env bash
# Build the fitcoach_cpp pybind11 extension into cpp/build/.
# Run from anywhere; assumes the `fitcoach` conda env is active.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_EXE="$(python -c 'import sys; print(sys.executable)')"
PYBIND11_DIR="$(python -c 'import pybind11; print(pybind11.get_cmake_dir())')"

cmake -S cpp -B cpp/build \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$PYTHON_EXE" \
  -Dpybind11_DIR="$PYBIND11_DIR"

cmake --build cpp/build --parallel

echo
echo "Built. Quick sanity check:"
PYTHONPATH="cpp/build:${PYTHONPATH:-}" python -c \
  "import fitcoach_cpp; print('joint_angle(90°) =', fitcoach_cpp.joint_angle((0.0,1.0),(0.0,0.0),(1.0,0.0)))"
