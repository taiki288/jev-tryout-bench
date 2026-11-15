#!/usr/bin/env python3
"""Time per decision over a grid of option counts x image sizes (any width x height, e.g. 2:1 panoramas).

Uses the same backends as eval_acc.py (bench/adapters.py) and the support-ticket question of bench2.py
(up to 100 options). Only the time and the input token count are of interest here, not the answer.

  python3 bench/latency_grid.py --backend systemone --url http://127.0.0.1:8080/v1/systemone --model jev \
      --options 10,20,50,100 --image 0,448x224,896x448,1344x672,1920x960 --label d1-2b-bf16-llamacpp \
      --out results/grid_d1-2b_llamacpp_bf16
Image sizes: 0 = no image, N = N x N, WxH. Combinations a model cannot take (Smaug: more than 20 options)
are skipped. Writes <out>.grid.jsonl (one line per combination: mean / median / p90 ms, input tokens).
"""
import argparse, json, pathlib, statistics as st, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from adapters import BACKENDS  # noqa: E402
from bench2 import TICKETS, make_record  # noqa: E402


def panorama(w, h):
    """A synthetic street-like picture (content does not change the time, only the size does)."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (w, h), (135, 180, 230))
    d = ImageDraw.Draw(img)
    d.rectangle([0, h // 2, w, h], fill=(90, 90, 90))
    for x in range(0, w, max(8, w // 24)):
        d.rectangle([x, h // 3, x + w // 48, h // 2], fill=((x * 7) % 255, 120, 80))
        d.line([(x, h), (w // 2, h // 2)], fill=(230, 230, 230), width=2)
    return img


def parse_size(s):
    if s == "0":
        return None
    w, _, h = s.partition("x")
    return int(w), int(h or w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=list(BACKENDS), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--url", default="http://localhost:8000/v1/systemone")
    ap.add_argument("--image-raw", action="store_true")
    ap.add_argument("--code-repo", default=None)
    ap.add_argument("--bnb", choices=["int8", "nf4"], default=None)
    ap.add_argument("--engine-dir", default=None)
    ap.add_argument("--options", default="10,20,50,100")
    ap.add_argument("--image", default="0,448x224,896x448,1344x672,1920x960")
    ap.add_argument("--reps", type=int, default=5, help="questions per combination (different tickets)")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    be = BACKENDS[a.backend](a)
    sizes = [parse_size(s) for s in a.image.split(",")]
    imgs = {s: panorama(*s) for s in sizes if s}
    out = open(a.out + ".grid.jsonl", "a")
    print(f"{a.label}: options x image -> mean ms (input tokens)", flush=True)
    for k in [int(x) for x in a.options.split(",")]:
        for s in sizes:
            img = imgs.get(s)
            try:
                for i in range(a.warmup):
                    be.ask(make_record(i % len(TICKETS), k)[0], img)
            except Exception as e:  # e.g. Smaug's 20-option limit, or a prompt longer than the engine allows
                print(f"  K={k:3d} img={s}: skipped ({type(e).__name__}: {str(e)[:120]})", flush=True)
                continue
            ms, toks = [], []
            for i in range(a.reps):
                rec, _ = make_record(i % len(TICKETS), k)
                t = time.perf_counter()
                _, n = be.ask(rec, img)
                ms.append((time.perf_counter() - t) * 1000); toks.append(n or 0)
            row = {"label": a.label, "options": k, "width": s[0] if s else 0, "height": s[1] if s else 0,
                   "mean_ms": round(st.mean(ms), 1), "median_ms": round(st.median(ms), 1),
                   "p90_ms": round(sorted(ms)[int(0.9 * (len(ms) - 1))], 1), "input_tokens": round(st.mean(toks))}
            out.write(json.dumps(row) + "\n"); out.flush()
            print(f"  K={k:3d} img={'%dx%d' % s if s else 'none':>9}: {row['mean_ms']:7.1f} ms "
                  f"({row['input_tokens']} tok)", flush=True)


if __name__ == "__main__":
    main()
