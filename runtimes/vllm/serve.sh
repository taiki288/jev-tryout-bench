#!/bin/bash
# Start a vLLM server (port 8000) for an already-downloaded HF repo.
# build the patched image once:  docker build -t clef-vllm:0.32.0-patched runtimes/vllm
#   (patch_embedding.py: bf16 LM-head gather for AutoRound Clef-omni + lifts the 26-option cap of the joint schema path)
# usage: runtimes/vllm/serve.sh <name> <hf repo> [vllm args]
# example (Clef-omni INT4):
#   runtimes/vllm/serve.sh clef-omni-int4 moeshawky/clef-omni-int4-autoround --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.55
# stop: docker rm -f <name>
. "$(dirname "$0")/../common.sh"
set -e
name=$1; repo=$2; shift 2
snap=$(snapshot "$repo"); cpath=/root/.cache/huggingface/${snap#$HF_DIR/}
mkdir -p "$ROOT/runtimes/vllm/cache"
docker rm -f "$name" >/dev/null 2>&1 || true
docker run -d --name "$name" --gpus all --ipc=host -p 8000:8000 \
  -v "$HF_DIR":/root/.cache/huggingface -v "$ROOT/runtimes/vllm/cache":/root/.cache/vllm -e HF_HUB_OFFLINE=1 \
  clef-vllm:0.32.0-patched "$cpath" --served-model-name "$name" --host 0.0.0.0 --port 8000 "$@" >/dev/null
guard_container "$name" "$ROOT/results/$name.guard.log"
echo "started $name; wait: docker logs -f $name  (ready at 'Application startup complete')"
