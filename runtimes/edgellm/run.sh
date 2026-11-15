#!/bin/bash
# Accuracy set (bench/eval_acc.py) on a TensorRT Edge-LLM engine built by runtimes/edgellm/build.sh.
# The engine runs in the benchmark process (no server), under the memory guard.
# usage: runtimes/edgellm/run.sh <name> <label> <out> [eval_acc.py options]
#   MODE=grid runtimes/edgellm/run.sh ...  runs bench/latency_grid.py (options x image size) instead
# example:
#   runtimes/edgellm/run.sh smaug-fp16 smaug-decisions-fp16-edgellm results/acc_smaug-decisions_edgellm_fp16 --options 10,20 --image 448
. "$(dirname "$0")/../common.sh"
EDGE_VENV="${EDGE_VENV:-$HOME/.venv-edgellm}"
name=$1; label=$2; out=$3; shift 3
cd "$ROOT"
script=bench/eval_acc.py; [ "${MODE:-acc}" = grid ] && script=bench/latency_grid.py
"$EDGE_VENV/bin/python" $script --backend edgellm --model "$HOME/edgellm-ckpt/$name" \
  --engine-dir "$HOME/edgellm-engines/$name" --label "$label" --out "$out" "$@" > "$out.log" 2>&1 &
pid=$!; guard_pid $pid "$out.log"; wait $pid; rc=$?
tail -20 "$out.log"; exit $rc
