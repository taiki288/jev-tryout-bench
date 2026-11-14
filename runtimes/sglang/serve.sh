#!/bin/bash
# Start an SGLang server (port 30000) with the Clef joint-head image (Qwen3.5 Clef / Clef-flash only).
# usage: runtimes/sglang/serve.sh <name> <hf repo> [sglang args]
# examples:
#   runtimes/sglang/serve.sh clef-flash Cloudflare/clef-flash --mem-fraction-static 0.55 --skip-server-warmup
#   runtimes/sglang/serve.sh clef-int4 Vishva007/clef-W4A16-AutoRound-GPTQ --mem-fraction-static 0.55 --watchdog-timeout 7200 --skip-server-warmup
# GPTQ Marlin kernels are JIT-compiled for sm_87 on the first request (~11 min); they are cached in
# runtimes/sglang/cache, so keep --skip-server-warmup and a long --watchdog-timeout for the first run.
# stop: docker rm -f <name>
. "$(dirname "$0")/../common.sh"
set -e
name=$1; repo=$2; shift 2
snap=$(snapshot "$repo"); cpath=/root/.cache/huggingface/${snap#$HF_DIR/}
mkdir -p "$ROOT/runtimes/sglang/cache"
docker rm -f "$name" >/dev/null 2>&1 || true
docker run -d --name "$name" --gpus all --ipc=host --shm-size 16g -p 30000:30000 \
  -v "$HF_DIR":/root/.cache/huggingface -v "$ROOT/runtimes/sglang/cache":/sglang-cache \
  -e SGLANG_JIT_CACHE_DIR=/sglang-cache/jit -e HF_HUB_OFFLINE=1 --entrypoint sglang \
  lmsysorg/sglang:dev-cu13-clef serve --model-path "$cpath" --served-model-name "$name" \
  --host 0.0.0.0 --port 30000 "$@" >/dev/null
guard_container "$name" "$ROOT/results/$name.guard.log"
echo "started $name; wait: docker logs -f $name  (ready at 'Uvicorn running')"
