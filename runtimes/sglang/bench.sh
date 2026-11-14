#!/bin/bash
# Wait for the SGLang server <name> and run the benchmark against it.
# usage: runtimes/sglang/bench.sh <name> <label> <out> [bench_vllm options]
. "$(dirname "$0")/../common.sh"
n=$1; label=$2; out=$3; shift 3
cd "$ROOT"
wait_container_log "$n" "Uvicorn running" || exit 1
"$PY" bench/bench_vllm.py --url http://localhost:30000/v1/systemone --model "$n" --label "$label" --out "$out" "$@"
