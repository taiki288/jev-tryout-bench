#!/bin/bash
# Create <repo>/.venv for the in-process (host) benchmark and the benchmark clients.
# Versions are the ones measured in docs/RESULTS_clef.md (Jetson AGX Orin, L4T R39.2, CUDA 13.2).
#   torch 2.14.1+cu130 (sm_80 code, runs on sm_87), transformers 5.20.1,
#   flash-linear-attention 0.6.0 + causal-conv1d 1.7.0 (fast path for Qwen3.5 Gated DeltaNet),
#   gptqmodel 7.3.4 (Marlin INT4/INT8 kernels), bitsandbytes 0.50.2 (NF4; INT8 does not work on Orin)
# Needs uv (https://docs.astral.sh/uv/) and the CUDA toolkit in /usr/local/cuda. Builds take ~1 h on Orin.
set -ex
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.7 MAX_JOBS=8
uv venv -q --python 3.12 .venv
uv pip install --python .venv "torch==2.14.1" "torchvision==0.29.1" --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv "transformers==5.20.1" "huggingface_hub==1.33.0" "accelerate==1.15.0" "safetensors==0.8.0" \
  pillow av "flash-linear-attention==0.6.0" "auto-round==0.16.1" "bitsandbytes==0.50.2" setuptools wheel ninja packaging
CAUSAL_CONV1D_FORCE_BUILD=TRUE uv pip install --python .venv --no-build-isolation "causal-conv1d==1.7.0"
uv pip freeze --python .venv | grep -E "^(torch|transformers)==" > /tmp/jev_pins.txt
uv pip install --python .venv --no-build-isolation "gptqmodel==7.3.4" -c /tmp/jev_pins.txt
.venv/bin/python -c "import torch, fla, causal_conv1d, gptqmodel; print(torch.__version__, torch.cuda.is_available())"
