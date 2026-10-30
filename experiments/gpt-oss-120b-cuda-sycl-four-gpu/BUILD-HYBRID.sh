#!/usr/bin/env bash
# Reproduce the tested CUDA+SYCL llama.cpp build (source revision 52ffa02).
# Assumes Intel oneAPI 2026.1 + NVIDIA CUDA toolkit installed.
# Do not run on an existing production build directory.
set -euo pipefail
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
SOURCE_DIR="${SOURCE_DIR:-$HOME/ornith-iq3-test/llama.cpp}"
BUILD_DIR="${BUILD_DIR:-$PWD/build-cuda-sycl}"
cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" -G Ninja \
 -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_C_COMPILER=/opt/intel/oneapi/compiler/2026.1/bin/icx \
 -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/2026.1/bin/icpx \
 -DGGML_CUDA=ON -DGGML_SYCL=ON -DGGML_SYCL_F16=ON \
 -DGGML_SYCL_DNN=OFF -DGGML_CUDA_FA=ON
cmake --build "$BUILD_DIR" --target llama-cli llama-tokenize -j 8
"$BUILD_DIR/bin/llama-cli" --list-devices
# Verify four reported devices: CUDA0/CUDA1/SYCL0/SYCL1.
# No source-level kernel modifications were required for this build.