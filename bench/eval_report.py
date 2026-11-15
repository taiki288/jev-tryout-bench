#!/usr/bin/env python3
"""Markdown table from eval_acc.py results (one row per run label).

  python3 bench/eval_report.py results/acc_*.acc.jsonl --options 20 --image 448 [--ref clef-omni-int4-vllm]

Columns: accuracy per task, overall accuracy (all items weighted equally), mean probability given to
the right answer (separates models even when both are right), mean time for text and
image items, and (with --ref) how often the answer equals the reference run's answer.
"""
import argparse, json, statistics as st
from collections import defaultdict

TASKS = ["intent", "shape", "count", "word", "error", "scene", "receipt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--options", type=int, default=20)
    ap.add_argument("--image", type=int, default=448, help="image size for the image items")
    ap.add_argument("--ref", default=None, help="label of the reference run for the agreement column")
    a = ap.parse_args()

    runs = defaultdict(dict)  # label -> (item) -> row
    for f in a.files:
        for line in open(f):
            r = json.loads(line)
            if r["options"] == a.options and r["image"] in (0, a.image):
                runs[r["label"]][r["item"]] = r
    ref = runs.get(a.ref, {})
    head = ["run"] + TASKS + ["overall", "mean p(gold)", "ms (text)", "ms (image)"] + (["agree w/ ref"] if a.ref else [])
    print(f"K={a.options}, image {a.image}px\n")
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" + "---:|" * (len(head) - 1))
    for label, rows in runs.items():
        cells = [label]
        for t in TASKS:
            rs = [r for r in rows.values() if r["task"] == t]
            cells.append(f"{sum(r['correct'] for r in rs)}/{len(rs)}" if rs else "–")
        cells.append(f"{st.mean(r['correct'] for r in rows.values()):.1%}")
        cells.append(f"{st.mean(r['p_gold'] for r in rows.values()):.3f}")
        for img in (False, True):
            ms = [r["lat_ms"] for r in rows.values() if (r["image"] > 0) == img]
            cells.append(f"{st.mean(ms):.0f}" if ms else "–")
        if a.ref:
            common = [i for i in rows if i in ref]
            cells.append(f"{st.mean(rows[i]['choice'] == ref[i]['choice'] for i in common):.1%}" if common else "–")
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
