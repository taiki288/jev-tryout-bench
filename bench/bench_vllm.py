#!/usr/bin/env python3
"""Same benchmark as bench2.py, but against a running server's POST /v1/systemone
(vLLM / SGLang). Same questions, options, image, reps; same output format
(<out>.runs.jsonl / <out>.summary.jsonl) so summarize2.py and export_dataset.py work.

Memory: the server is another process, so there is no torch peak; RAM comes from tegrastats
(system-wide, includes the server's preallocated KV cache).
"""
import argparse, base64, io, json, statistics as st, sys, time, urllib.request

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from bench import Tegra, test_image  # noqa: E402
from bench2 import make_record, pct  # noqa: E402


def post(url, body, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/v1/systemone")
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--options", default="10,50,100")
    ap.add_argument("--image", default="0,1024")
    ap.add_argument("--n-questions", type=int, default=10)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--ram-baseline-mb", type=int, default=None, help="RAM before the server started")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--image-raw", action="store_true", help="send images as raw base64 (ollama), not data URLs")
    a = ap.parse_args()

    tg = Tegra(); time.sleep(1.0)
    meta = {"model": a.model, "label": a.label, "server": a.url, "ram_before_mb": a.ram_baseline_mb,
            "ram_loaded_mb": tg.ram_now()}
    runs_f = open(a.out + ".runs.jsonl", "a")
    sum_f = open(a.out + ".summary.jsonl", "a")

    for img_size in [int(x) for x in a.image.split(",")]:
        img_url = None
        if img_size:
            buf = io.BytesIO(); test_image(img_size).save(buf, format="PNG")
            img_url = base64.b64encode(buf.getvalue()).decode()
            if not a.image_raw:
                img_url = "data:image/png;base64," + img_url
        for k in [int(x) for x in a.options.split(",")]:
            cfg = {**meta, "options": k, "image": img_size}
            try:
                bodies = []
                for i in range(a.n_questions):
                    rec, gold = make_record(i, k, None)
                    body = {"model": a.model, "state": rec["state"], "questions": rec["questions"]}
                    if img_url:
                        body["images"] = [img_url]
                    bodies.append((body, gold, list(rec["questions"]["route"]["criteria"])))

                def run(body):
                    t = time.perf_counter(); r = post(a.url, body); return time.perf_counter() - t, r

                for i in range(a.warmup):
                    run(bodies[i % len(bodies)][0])
                per_q, all_lat = [], []
                t_start = time.perf_counter()
                for qi, (body, gold, option_ids) in enumerate(bodies):
                    lat, probs_list, argmaxes, n_tok = [], [], [], None
                    for r_i in range(a.reps):
                        dt, resp = run(body)
                        ans = resp["answers"]["route"]
                        pr = ans["probabilities"]
                        probs = [float(pr[o]) for o in option_ids]
                        am = ans["choice"]
                        n_tok = resp["usage"]["input_tokens"]
                        lat.append(dt * 1000); probs_list.append(probs); argmaxes.append(am)
                        runs_f.write(json.dumps({"label": a.label, "model": a.model, "options": k, "image": img_size,
                                                 "question": qi, "rep": r_i, "lat_ms": dt * 1000, "argmax": am,
                                                 "gold": gold, "correct": am == gold, "p_gold": pr[gold],
                                                 "probs": [round(p, 6) for p in probs]}) + "\n")
                    mode = max(set(argmaxes), key=argmaxes.count)
                    spread = max(max(c) - min(c) for c in zip(*probs_list))
                    per_q.append({"question": qi, "input_tokens": n_tok, "lat_mean_ms": st.mean(lat),
                                  "lat_std_ms": st.pstdev(lat), "lat_cv": st.pstdev(lat) / st.mean(lat),
                                  "argmax_agree": argmaxes.count(mode) / len(argmaxes),
                                  "prob_spread_max": spread,
                                  "correct_rate": sum(x == gold for x in argmaxes) / len(argmaxes)})
                    all_lat += lat
                    print(f"  K={k} img={img_size} q{qi}: {st.mean(lat):.0f}±{st.pstdev(lat):.1f} ms tok={n_tok} "
                          f"agree={per_q[-1]['argmax_agree']:.2f} correct={per_q[-1]['correct_rate']:.1f}", flush=True)
                t_end = time.perf_counter()
                time.sleep(0.25)
                tsum = tg.summarize(t_start, t_end)
                rails_j = [v for kk, v in tsum.items() if kk.endswith("_energy_J")]
                rails_mw = [v for kk, v in tsum.items() if kk.endswith("_mean_mW")]
                n = len(all_lat)
                res = {**cfg, "status": "ok", "n_runs": n,
                       "input_tokens_mean": st.mean(q["input_tokens"] for q in per_q),
                       "lat_mean_ms": st.mean(all_lat), "lat_p50_ms": pct(all_lat, .5), "lat_p95_ms": pct(all_lat, .95),
                       "lat_min_ms": min(all_lat), "lat_max_ms": max(all_lat),
                       "jitter_within_std_ms": st.mean(q["lat_std_ms"] for q in per_q),
                       "jitter_within_cv": st.mean(q["lat_cv"] for q in per_q),
                       "lat_std_overall_ms": st.pstdev(all_lat),
                       "argmax_agree": st.mean(q["argmax_agree"] for q in per_q),
                       "prob_spread_max": max(q["prob_spread_max"] for q in per_q),
                       "accuracy": st.mean(q["correct_rate"] for q in per_q),
                       "torch_peak_gib": None,
                       "total_mean_W": sum(rails_mw) / 1000 if rails_mw else None,
                       "energy_per_req_J": sum(rails_j) / n if rails_j else None,
                       **tsum, "per_question": per_q}
            except urllib.error.HTTPError as e:
                res = {**cfg, "status": "error", "error": f"HTTP {e.code}: {e.read().decode()[:300]}"}
            except Exception as e:
                res = {**cfg, "status": "error", "error": f"{type(e).__name__}: {e}"[:300]}
            print(json.dumps({kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in res.items()
                              if kk not in meta and kk != "per_question"}), flush=True)
            sum_f.write(json.dumps(res) + "\n"); sum_f.flush(); runs_f.flush()
    tg.stop()


if __name__ == "__main__":
    main()
