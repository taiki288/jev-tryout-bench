#!/bin/bash
# Start the ollama container, load <model> (timed), run the benchmark, stop the container.
# usage: runtimes/ollama/run.sh <ollama model> <label> <out> [bench_vllm options]
# example: runtimes/ollama/run.sh clef-flash ollama-q8_0 results/clef-flash_ollama_q8_0
# Models live in runtimes/ollama/models (pull: docker exec jev-ollama ollama pull <model>).
# ollama takes images as raw base64 (not data URLs), hence --image-raw.
. "$(dirname "$0")/../common.sh"
model=$1; label=$2; out=$3; shift 3
cd "$ROOT"; name=jev-ollama
used0=$(mem_used); mkdir -p runtimes/ollama/models
docker rm -f $name >/dev/null 2>&1
docker run -d --name $name --gpus all --ipc=host -p 11434:11434 -v "$ROOT/runtimes/ollama/models":/root/.ollama \
  -e OLLAMA_KEEP_ALIVE=-1 ollama/ollama:latest >/dev/null
guard_container $name "${out}.server.log"
until curl -sf localhost:11434/api/version >/dev/null; do sleep 1; done
docker exec $name ollama list | grep -q "^$model" || docker exec $name ollama pull "$model"
t0=$(date +%s.%N)
curl -s localhost:11434/v1/systemone -H "Content-Type: application/json" -d "{\"model\":\"$model\",\"state\":\"I cannot log in\",\"questions\":{\"route\":{\"type\":\"choice\",\"instructions\":\"Which team?\",\"criteria\":{\"billing\":null,\"login\":null}}}}" >/dev/null
echo "load + first request: $(echo "$(date +%s.%N) - $t0" | bc) s"
"$PY" bench/bench_vllm.py --url http://127.0.0.1:11434/v1/systemone --model "$model" --image-raw \
  --label "$label" --out "$out" --ram-baseline-mb "$used0" "$@"
docker logs $name > "${out}.server.log" 2>&1
docker rm -f $name >/dev/null
