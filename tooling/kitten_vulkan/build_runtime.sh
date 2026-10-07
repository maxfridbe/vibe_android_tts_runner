#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TASK_DIR="$ROOT/output/kitten-vulkan"
export PATH="$TASK_DIR:$PATH"
cmake -S "$TASK_DIR/executorch" -B "$TASK_DIR/runtime-build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE="$TASK_DIR/venv/bin/python" \
  -DEXECUTORCH_BUILD_VULKAN=ON \
  -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON \
  -DEXECUTORCH_ENABLE_BUNDLE_IO=ON \
  -DEXECUTORCH_BUILD_DEVTOOLS=ON \
  -DEXECUTORCH_BUILD_XNNPACK=OFF \
  -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=OFF \
  -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=OFF \
  -DEXECUTORCH_BUILD_TESTS=OFF
cmake --build "$TASK_DIR/runtime-build" --target executor_runner -j "${BUILD_JOBS:-4}"
