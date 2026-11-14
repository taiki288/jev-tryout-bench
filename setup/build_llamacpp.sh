#!/bin/bash
# Build llama.cpp with CUDA for Orin (sm_87). The native `clef` architecture and /v1/systemone are upstream.
# Measured with commit a0ad695. Do not build while a benchmark is running (CPU load skews timings).
set -ex
LLAMA_CPP="${LLAMA_CPP:-$HOME/llama.cpp}"
[ -d "$LLAMA_CPP" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_CPP"
cd "$LLAMA_CPP"
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build -j 10 --target llama-server llama-cli
