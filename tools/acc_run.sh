#!/bin/bash
# Run the accuracy set (dataset/eval_v1) on every model in a list file, one after another.
#
# usage: tools/acc_run.sh <list> [download] [run] [name ...]
#   <list>     a file in tools/acc_lists/ (format below)
#   download   fetch the weights in the list (do this before measuring: no downloads during a run)
#   run        measure every line whose results/acc_<name>_<runtime>_<precision>.acc.jsonl does not exist yet
#   name ...   only the lines with these model names (default: all)
#   (with neither download nor run: both)
# examples:
#   tools/acc_run.sh tools/acc_lists/small.txt download
#   tools/acc_run.sh tools/acc_lists/small.txt run d1-2b
#   ACC_OPTS="--options 20 --image 448 --limit 3" tools/acc_run.sh tools/acc_lists/quant.txt run   # quick try
#
# List format: one run per line, fields separated by spaces, "#" starts a comment.
#   opts <eval_acc.py options>                         options for every run in the file (ACC_OPTS overrides)
#   <runtime> <name> <precision> <repo> [<file> <mmproj>]
#     runtime    host:<backend>  transformers in this process (backend: clef, intern, smaug; see bench/adapters.py)
#                llamacpp        llama-server with <file> (GGUF) + <mmproj> (runtimes/llamacpp/run.sh)
#                edgellm         TensorRT Edge-LLM engine named <file> (build it first: runtimes/edgellm/build.sh)
#     mmproj     a file in <repo>, or <other repo>:<file>
#     extra eval_acc.py options for one line can follow after "--", e.g.  ... -- --bnb int8
# To redo a run, delete its results/acc_*.acc.jsonl (rows are appended, not replaced).
. "$(dirname "$0")/../runtimes/common.sh"
cd "$ROOT"
list=$1; shift || true
[ -f "$list" ] || { sed -n '2,25p' "$0"; exit 1; }
steps=""; names=()
for x in "$@"; do case $x in download|run) steps+=" $x" ;; *) names+=("$x") ;; esac; done
steps=${steps:-download run}

opts=$(awk '$1 == "opts" { $1 = ""; print; exit }' "$list")
read -ra OPTS <<< "${ACC_OPTS:-$opts}"
HF="$ROOT/.venv/bin/hf"

# lines of the list: runtime name precision repo file mmproj [-- extra options]
entries() { grep -vE '^\s*(#|$|opts\s)' "$list" | sed 's/#.*//'; }
wanted() { [ ${#names[@]} -eq 0 ] && return 0; local n; for n in "${names[@]}"; do [ "$n" = "$1" ] && return 0; done; return 1; }

if [[ $steps == *download* ]]; then
  while read -r rt name prec repo file mm _; do
    wanted "$name" || continue
    case $rt in
      host:*)   "$HF" download "$repo" >/dev/null && echo "downloaded $repo" ;;
      llamacpp) "$HF" download "$repo" "$file" >/dev/null && echo "downloaded $repo $file"
                if [[ $mm == *:* ]]; then "$HF" download "${mm%%:*}" "${mm#*:}" >/dev/null
                else "$HF" download "$repo" "$mm" >/dev/null; fi ;;
      edgellm)  "$HF" download "$repo" >/dev/null && echo "downloaded $repo (build the engine: runtimes/edgellm/build.sh $repo $file)" ;;
      *) echo "unknown runtime: $rt" ;;
    esac
  done < <(entries)
fi

if [[ $steps == *run* ]]; then
  while read -r line; do
    read -r rt name prec repo file mm _ <<< "$line"
    wanted "$name" || continue
    extra=(); [[ $line == *" -- "* ]] && read -ra extra <<< "${line#* -- }"
    runtime=${rt%%:*}
    out=results/acc_${name}_${runtime}_${prec}; label=$name-$prec-$runtime
    [ -s "$out.acc.jsonl" ] && { echo "skip $label (done)"; continue; }
    echo "$(date +%T) start $label"
    case $rt in
      host:*)
        "$PY" bench/eval_acc.py --backend "${rt#host:}" --model "$repo" --label "$label" --out "$out" \
          "${OPTS[@]}" "${extra[@]}" > "$out.log" 2>&1 < /dev/null &
        pid=$!; guard_pid $pid "$out.log"; wait $pid ;;
      llamacpp)
        if [[ $mm == *:* ]]; then mm="$(snapshot "${mm%%:*}")/${mm#*:}" || continue; fi
        MODE=acc runtimes/llamacpp/run.sh "$repo" "$file" "$mm" "$label" "$out" "${OPTS[@]}" "${extra[@]}" \
          > "$out.log" 2>&1 < /dev/null ;;
      edgellm)
        runtimes/edgellm/run.sh "$file" "$label" "$out" "${OPTS[@]}" "${extra[@]}" > /dev/null < /dev/null ;;
      *) echo "unknown runtime: $rt"; continue ;;
    esac
    echo "$(date +%T) $label: exit $?"
  done < <(entries)
fi
