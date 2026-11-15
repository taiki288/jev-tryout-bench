#!/usr/bin/env python3
"""Accuracy run on the eval_v1 question set (bench/evalset.py), with the time of each answer.

Every item is asked once per option count K; image items are asked once per image size.
Writes <out>.acc.jsonl (one line per answer) and prints accuracy per task. Compare runs with
bench/eval_report.py.

Backends:
  clef       transformers in this process, official Cloudflare loader code (like bench2.py)
  systemone  a running server's POST /v1/systemone (vLLM / SGLang / llama.cpp / ollama)

Examples:
  .venv/bin/python bench/eval_acc.py --backend clef --model Cloudflare/clef-flash \
      --label clef-flash-bf16 --out results/acc_clef-flash_host_bf16
  python3 bench/eval_acc.py --backend systemone --url http://localhost:8000/v1/systemone \
      --model clef-omni-int4 --label clef-omni-int4-vllm --out results/acc_clef-omni_vllm_int4
Options: --options 10,20 (max 30)  --image 448 (sizes for image items, e.g. 224,448,1024)
         --tasks intent,shape,count,word,error,scene,receipt  --limit N (first N items per task, for a quick try)
"""
import argparse, json, pathlib, statistics as st, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from evalset import load_items, make_question  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


from adapters import BACKENDS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=list(BACKENDS), required=True)
    ap.add_argument("--model", required=True, help="weights repo (clef) or served model name (systemone)")
    ap.add_argument("--url", default="http://localhost:8000/v1/systemone")
    ap.add_argument("--image-raw", action="store_true", help="send images as raw base64 (ollama)")
    ap.add_argument("--code-repo", default=None)
    ap.add_argument("--bnb", choices=["int8", "nf4"], default=None)
    ap.add_argument("--eval-dir", default=str(ROOT / "dataset" / "eval_v1"))
    ap.add_argument("--options", default="10,20")
    ap.add_argument("--image", default="448", help="image sizes for the image items")
    ap.add_argument("--tasks", default="intent,shape,count,word,error,scene,receipt")
    ap.add_argument("--limit", type=int, default=0, help="only the first N items of each task")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from PIL import Image
    items, tasks = load_items(a.eval_dir)
    want = a.tasks.split(",")
    items = [it for it in items if it["task"] in want]
    if a.limit:
        items = [it for t in want for it in [x for x in items if x["task"] == t][:a.limit]]
    t0 = time.perf_counter()
    be = BACKENDS[a.backend](a)
    print(f"{a.label}: backend ready in {time.perf_counter() - t0:.1f}s, {len(items)} items", flush=True)

    sizes = [int(x) for x in a.image.split(",")]
    imgs = {}
    for it in items:
        if it["image"]:
            m = Image.open(pathlib.Path(a.eval_dir) / it["image"]).convert("RGB")
            for s in sizes:
                imgs[it["id"], s] = m if s == m.width else m.resize((s, s), Image.LANCZOS)

    jobs = []  # (item, K, image size)
    for k in [int(x) for x in a.options.split(",")]:
        for it in items:
            for s in (sizes if it["image"] else [0]):
                jobs.append((it, k, s))
    for it, k, s in jobs[:a.warmup]:
        be.ask(make_question(it, tasks, k)[0], imgs.get((it["id"], s)))

    out = open(a.out + ".acc.jsonl", "a")
    res = {}
    for n, (it, k, s) in enumerate(jobs):
        rec, gold = make_question(it, tasks, k)
        t = time.perf_counter()
        probs, ntok = be.ask(rec, imgs.get((it["id"], s)))
        dt = (time.perf_counter() - t) * 1000
        crit = rec["questions"]["route"]["criteria"]
        choice = max(probs, key=probs.get)
        top = sorted(probs, key=probs.get, reverse=True)[:3]
        row = {"label": a.label, "model": a.model, "backend": a.backend, "item": it["id"], "task": it["task"],
               "options": k, "image": s, "lat_ms": round(dt, 2), "input_tokens": ntok,
               "choice": crit[choice], "gold": it["gold"], "correct": choice == gold,
               "p_gold": round(probs.get(gold, 0.0), 5), "top3": [[crit[o], round(probs[o], 5)] for o in top]}
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        res.setdefault((it["task"], k, s), []).append(row)
        if (n + 1) % 50 == 0:
            out.flush(); print(f"  {n + 1}/{len(jobs)}", flush=True)
    out.close()
    print(f"\n{a.label}  (task, K, image): accuracy, mean ms, mean tokens")
    for (task, k, s), rows in sorted(res.items()):
        toks = [r["input_tokens"] for r in rows if r["input_tokens"]]
        print(f"  {task:7s} K={k:3d} img={s:4d}: {sum(r['correct'] for r in rows):3d}/{len(rows):3d} "
              f"= {st.mean(r['correct'] for r in rows):.2f}  {st.mean(r['lat_ms'] for r in rows):7.1f} ms"
              f"  {st.mean(toks) if toks else 0:6.0f} tok")


if __name__ == "__main__":
    main()
