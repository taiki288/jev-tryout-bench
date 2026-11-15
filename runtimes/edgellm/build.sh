#!/bin/bash
# Build a TensorRT Edge-LLM engine (text + vision) for a decision model, with Edge-LLM's direct builder.
# Setup (once, see setup/README.md):  uv venv --system-site-packages ~/.venv-edgellm
#                                     uv pip install "tensorrt-edgellm[runtime,onnx]==0.12.0"
# usage: runtimes/edgellm/build.sh <hf repo or local checkpoint dir> <name> [trtedge build options]
#   -> checkpoint ~/edgellm-ckpt/<name> (links to the HF cache, config.json patched if needed)
#      engines    ~/edgellm-engines/<name>, log ~/edgellm-engines/<name>.build.log
# example:
#   runtimes/edgellm/build.sh abacusai/Smaug-Decisions smaug-fp16 --max-image-tokens 560 --max-image-tokens-per-image 560
# INT8 (W8A8 SmoothQuant): quantize first (needs: uv pip install "tensorrt-edgellm[quantize]==0.12.0"), then build from the output dir
#   ~/.venv-edgellm/bin/trtedge quantize llm --model-dir ~/edgellm-ckpt/smaug-fp16 --output-dir ~/edgellm-ckpt/smaug-int8sq-src \
#       --quantization int8_sq --dtype fp16 --num-samples 128
#   runtimes/edgellm/build.sh ~/edgellm-ckpt/smaug-int8sq-src smaug-int8sq --max-image-tokens 560 --max-image-tokens-per-image 560
# Then measure with runtimes/edgellm/run.sh <name> ... (or a line in tools/acc_lists/tensorrt.txt).
# Do not build while a benchmark is running.
. "$(dirname "$0")/../common.sh"
EDGE_VENV="${EDGE_VENV:-$HOME/.venv-edgellm}"
repo=$1; name=$2; shift 2
if [ -d "$repo" ]; then snap=$(readlink -f "$repo"); else snap=$(snapshot "$repo") || exit 1; fi
ckpt=$HOME/edgellm-ckpt/$name; eng=$HOME/edgellm-engines/$name
mkdir -p "$ckpt" "$HOME/edgellm-engines"
for f in "$snap"/*; do [ "$(basename "$f")" = config.json ] || ln -sfn "$(readlink -f "$f")" "$ckpt/$(basename "$f")"; done
# Newer transformers write Gemma 4's full-attention head size as text_config.per_layer_config.<layer>.head_dim;
# Edge-LLM 0.12 reads text_config.global_head_dim, and without it builds those layers with the wrong size.
python3 - "$snap/config.json" "$ckpt/config.json" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))
t = c.get("text_config", c)
dims = {v["head_dim"] for v in (t.get("per_layer_config") or {}).values() if "head_dim" in v}
if dims and not t.get("global_head_dim"):
    assert len(dims) == 1, dims
    t["global_head_dim"] = dims.pop()
    print("config.json: added global_head_dim", t["global_head_dim"])
json.dump(c, open(sys.argv[2], "w"), indent=2)
EOF
rm -rf "$eng"
cd /tmp   # outside any source checkout (Edge-LLM's advice for the wheel)
"$EDGE_VENV/bin/trtedge" build --model-dir "$ckpt" --engine-dir "$eng" --components llm,visual \
  --max-input-len 2048 --max-kv-cache-capacity 2112 --max-batch-size 1 "$@" > "$eng.build.log" 2>&1 &
pid=$!; guard_pid $pid "$eng.build.log"; t0=$(date +%s); wait $pid; rc=$?
echo "build exit $rc in $(( $(date +%s) - t0 )) s  (log: $eng.build.log)"
exit $rc
