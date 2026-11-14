#!/bin/bash
# Same in-process benchmark as runtimes/host/run.sh, inside the NVIDIA PyTorch container.
# build once:  docker build -t clef-torch:26.10 runtimes/docker
# usage: runtimes/docker/run.sh --model <hf repo> --label <label> --out results/<name> [bench2 options]
# The run is detached; follow it with: docker logs -f jev-docker-bench
. "$(dirname "$0")/../common.sh"
name=jev-docker-bench
docker rm -f $name >/dev/null 2>&1
# tegrastats only needs libc, so the host binary is mounted for power / memory sampling
docker run -d --name $name --gpus all --ipc=host \
  -v "$HF_DIR":/root/.cache/huggingface -v "$ROOT":/work -w /work \
  -v /usr/bin/tegrastats:/usr/bin/tegrastats:ro -e HF_HUB_OFFLINE=1 \
  -e NVPMODEL_Q="$(nvpmodel -q 2>/dev/null | head -1)" \
  clef-torch:26.10 python bench/bench2.py "$@" >/dev/null
guard_container $name "$ROOT/results/$name.guard.log"
echo "started $name (docker logs -f $name)"
