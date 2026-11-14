#!/usr/bin/env python3
"""Send benchmark questions from dataset/ to a running server's /v1/systemone and print the time.

Standard library only. Example:
  python3 try_request.py --url http://localhost:8000/v1/systemone --model clef-omni-int4 --k 100 --img 1024
Options:
  --k 10|50|100      number of options (dataset/records_K{k}_img{img}.json)
  --img 0|1024       0 = no image, 1024 = with dataset/images/test_1024.png
  --q 0              question index (0-9), or -1 for all 10
  --n 3              how many times to send each question
  --raw              send the image as raw base64 (ollama) instead of a data URL
"""
import argparse, base64, json, pathlib, statistics, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--model", default="clef")
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--img", type=int, default=0)
ap.add_argument("--q", type=int, default=0)
ap.add_argument("--n", type=int, default=3)
ap.add_argument("--raw", action="store_true")
a = ap.parse_args()

ds = pathlib.Path(__file__).resolve().parent.parent / "dataset"
recs = json.loads((ds / f"records_K{a.k}_img{a.img}.json").read_text())
img = None
if a.img:
    img = base64.b64encode((ds / f"images/test_{a.img}.png").read_bytes()).decode()
    if not a.raw:
        img = "data:image/png;base64," + img

qs = range(len(recs)) if a.q < 0 else [a.q]
all_t = []
for qi in qs:
    r = recs[qi]
    body = {"model": a.model, "state": r["state"], "questions": r["questions"]}
    if img:
        body["images"] = [img]
    data = json.dumps(body).encode()
    for i in range(a.n):
        t = time.perf_counter()
        req = urllib.request.Request(a.url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            out = json.loads(resp.read())
        dt = (time.perf_counter() - t) * 1000
        ans = out["answers"]["route"]
        crit = r["questions"]["route"]["criteria"]
        print(f"q{qi} #{i + 1}: {dt:8.1f} ms  tokens={out['usage']['input_tokens']}  "
              f"answer={ans['choice']} ({crit.get(ans['choice'])})  p={ans['probabilities'][ans['choice']]:.3f}")
        if i or len(qs) > 1:  # the first request of a single question may include warm-up
            all_t.append(dt)
if all_t:
    print(f"mean {statistics.mean(all_t):.1f} ms over {len(all_t)} requests (K={a.k}, image={a.img})")
