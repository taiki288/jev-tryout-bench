#!/bin/bash
# Start llama-server with a GGUF + mmproj, run the benchmark, stop the server.
# build llama.cpp once (see setup/README.md):  LLAMA_CPP=~/llama.cpp
# usage: runtimes/llamacpp/run.sh <hf repo> <gguf> <mmproj> <label> <out> [bench_vllm options]
#   <mmproj> is a file name in the same repo, or an absolute path
#   MODE=acc runtimes/llamacpp/run.sh ...   runs the accuracy set (bench/eval_acc.py) instead of the speed benchmark
# example:
#   runtimes/llamacpp/run.sh ggml-org/Clef-Flash-GGUF Clef-Flash-BF16.gguf mmproj-Clef-Flash-BF16.gguf llamacpp-bf16 results/clef-flash_llamacpp_bf16
# Clef evaluates the whole prompt in one batch, so -b/-ub must exceed the longest prompt (3.5k tokens here).
. "$(dirname "$0")/../common.sh"
LLAMA_CPP="${LLAMA_CPP:-$HOME/llama.cpp}"
repo=$1; gguf=$2; mm=$3; label=$4; out=$5; shift 5
snap=$(snapshot "$repo") || exit 1
cd "$ROOT"
used0=$(mem_used); log=${out}.server.log; t0=$(date +%s)
[[ $mm == /* ]] || mm="$snap/$mm"   # an absolute path takes the mmproj from another repo
"$LLAMA_CPP/build/bin/llama-server" -m "$snap/$gguf" --mmproj "$mm" -ngl 99 -c 8192 -b 8192 -ub 8192 \
  -np 1 --host 127.0.0.1 --port 8080 --alias jev > "$log" 2>&1 &
pid=$!; guard_pid $pid "$log"
until curl -sf localhost:8080/health >/dev/null; do kill -0 $pid 2>/dev/null || { echo SERVER DIED; tail -20 "$log"; exit 1; }; sleep 2; done
echo "ready in $(( $(date +%s)-t0 )) s"
if [ "${MODE:-speed}" = acc ]; then
  "$PY" bench/eval_acc.py --backend systemone --url http://127.0.0.1:8080/v1/systemone --model jev --label "$label" --out "$out" "$@"
else
  "$PY" bench/bench_vllm.py --url http://127.0.0.1:8080/v1/systemone --model jev --label "$label" --out "$out" --ram-baseline-mb "$used0" "$@"
fi
kill $pid; wait $pid 2>/dev/null
