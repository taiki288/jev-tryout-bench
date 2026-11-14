#!/bin/bash
# Quick check that every runtime script works on this machine (1 question, 3 reps, 10 options, no image / 448px).
# Runs each runtime one after another and stops its server; ~40 min in total (vLLM model load is the longest).
# Results go to results/try_* (git-ignored). Needs the models / images from setup/README.md.
# usage: tools/selftest.sh [runtime ...]   runtimes: llamacpp ollama docker sglang vllm   (default: all)
cd "$(dirname "$0")/.."
A="--options 10 --image 0,448 --n-questions 1 --reps 3 --warmup 1"
f() { grep -E "K=|ready|load|SERVER DIED|MEMGUARD|Traceback|\"status\": \"error" | cut -c1-150; }
run() {
  echo "=== $1 $(date +%T)"
  case $1 in
    host)     runtimes/host/run.sh --model Cloudflare/clef-flash $A --label try-host --out results/try_host 2>&1 | f ;;
    llamacpp) runtimes/llamacpp/run.sh ggml-org/Clef-Flash-GGUF Clef-Flash-BF16.gguf mmproj-Clef-Flash-BF16.gguf try-llama results/try_llamacpp $A 2>&1 | f ;;
    ollama)   runtimes/ollama/run.sh clef-flash try-ollama results/try_ollama $A 2>&1 | f ;;
    docker)   runtimes/docker/run.sh --model Cloudflare/clef-flash $A --label try-docker --out results/try_docker 2>&1 | f
              while docker inspect -f "{{.State.Running}}" jev-docker-bench 2>/dev/null | grep -q true; do sleep 5; done
              docker logs jev-docker-bench 2>&1 | f; docker rm -f jev-docker-bench >/dev/null ;;
    sglang)   runtimes/sglang/serve.sh clef-flash Cloudflare/clef-flash --mem-fraction-static 0.55 --skip-server-warmup
              runtimes/sglang/bench.sh clef-flash try-sglang results/try_sglang $A 2>&1 | f; docker rm -f clef-flash >/dev/null ;;
    vllm)     runtimes/vllm/serve.sh clef-omni-int4 moeshawky/clef-omni-int4-autoround --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.55
              runtimes/vllm/bench.sh clef-omni-int4 try-vllm results/try_vllm $A 2>&1 | f; docker rm -f clef-omni-int4 >/dev/null ;;
    *) echo "unknown runtime: $1" ;;
  esac
}
for r in "${@:-host llamacpp ollama docker sglang vllm}"; do for x in $r; do run $x; done; done
echo "SELFTEST_DONE $(date +%T)"
