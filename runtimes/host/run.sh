#!/bin/bash
# In-process benchmark with transformers on the host (no server).
# usage: runtimes/host/run.sh --model <hf repo> --label <label> --out results/<name> [bench2 options]
#   bench2 options: --options 10,50,100  --image 0,1024  --n-questions 10  --reps 10  --warmup 2
#                   --code-repo Cloudflare/<model> (loader code from the official repo when the weights repo is a quantized copy)
#                   --bnb int8|nf4 (bitsandbytes on the fly)
# example: runtimes/host/run.sh --model Cloudflare/clef-flash --label host-bf16 --out results/clef-flash_host_bf16
. "$(dirname "$0")/../common.sh"
cd "$ROOT"
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.7
"$PY" bench/bench2.py "$@" &
pid=$!; guard_pid $pid /dev/stderr; wait $pid
