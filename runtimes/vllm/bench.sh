#!/bin/bash
# Wait for the vLLM server <name> and run the benchmark against it.
# usage: runtimes/vllm/bench.sh <name> <label> <out> [bench_vllm options: --options --image --n-questions --reps]
. "$(dirname "$0")/../common.sh"
n=$1; label=$2; out=$3; shift 3
cd "$ROOT"
wait_container_log "$n" "Application startup complete" || exit 1
docker logs "$n" 2>&1 | grep -E "Loading weights took|Model loading took|KV cache memory|Using .*(Kernel|Experts)" | cut -c1-200
"$PY" bench/bench_vllm.py --url http://localhost:8000/v1/systemone --model "$n" --label "$label" --out "$out" "$@"
