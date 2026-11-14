# Shared helpers for the runtime scripts. Source it:  . "$(dirname "$0")/../common.sh"
# Settings (override with environment variables):
#   PY            python with the benchmark deps (default: <repo>/.venv/bin/python, see setup/)
#   HF_HOME       Hugging Face cache (default: ~/.cache/huggingface)
#   MIN_AVAIL_MB  memory guard: kill the job/container if MemAvailable drops below this (default 3000)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$ROOT/.venv/bin/python}"
HF_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
MIN_AVAIL_MB="${MIN_AVAIL_MB:-3000}"
mkdir -p "$ROOT/results"

mem_avail() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
mem_used()  { free -m | awk '/Mem/{print $3}'; }

# local snapshot dir of an HF repo (must be downloaded already: hf download <repo>)
snapshot() {
  local s; s=$(ls -d "$HF_DIR/hub/models--${1/\//--}/snapshots/"* 2>/dev/null | head -1)
  [ -n "$s" ] || { echo "not downloaded: $1  (run: hf download $1)" >&2; return 1; }
  echo "$s"
}

# Unified memory is shared by CPU and GPU: an OOM can hang the whole Jetson, so kill early.
guard_pid() {        # guard_pid <pid> <logfile>
  ( while kill -0 "$1" 2>/dev/null; do
      a=$(mem_avail); if [ "$a" -lt "$MIN_AVAIL_MB" ]; then echo "MEMGUARD: MemAvailable=${a}MB -> kill $1" >> "$2"; kill -9 "$1"; fi
      sleep 0.5; done ) >/dev/null 2>&1 &
}
guard_container() {  # guard_container <name> <logfile>
  ( while docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null | grep -q true; do
      a=$(mem_avail); if [ "$a" -lt "$MIN_AVAIL_MB" ]; then echo "MEMGUARD: MemAvailable=${a}MB -> docker kill $1" >> "$2"; docker kill "$1"; fi
      sleep 0.5; done ) >/dev/null 2>&1 &
}
# wait until a container log contains a pattern (fails if the container stops)
wait_container_log() {  # wait_container_log <name> <pattern>
  until docker logs "$1" 2>&1 | grep -q "$2"; do
    docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null | grep -q true || { echo "SERVER DIED"; docker logs "$1" 2>&1 | tail -30; return 1; }
    sleep 5
  done
}
