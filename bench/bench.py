#!/usr/bin/env python3
"""Clef family benchmark (transformers / on-host path).

Sweeps number of questions x number of options (x image on/off) and records
latency, memory and power per configuration. Power/RAM come from tegrastats,
sampled in the background and integrated over each timed run.

Output: one JSON object per configuration appended to --out (JSONL).
"""
import argparse, json, os, re, subprocess, sys, threading, time, statistics as st

import torch
from huggingface_hub import snapshot_download


# ---------------------------------------------------------------- tegrastats
class Tegra:
    RAIL = re.compile(r"(VDD_\w+|VIN_\w+)\s+(\d+)mW")
    RAM = re.compile(r"RAM (\d+)/(\d+)MB")
    GPU = re.compile(r"GR3D_FREQ (\d+)%")

    def __init__(self, interval_ms=100):
        self.samples = []  # (t, ram_mb, gr3d%, {rail: mW})
        self.p = subprocess.Popen(["tegrastats", "--interval", str(interval_ms)],
                                  stdout=subprocess.PIPE, text=True, bufsize=1)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            t = time.perf_counter()
            m = self.RAM.search(line); g = self.GPU.search(line)
            rails = {k: int(v) for k, v in self.RAIL.findall(line)}
            self.samples.append((t, int(m.group(1)) if m else None,
                                 int(g.group(1)) if g else None, rails))

    def window(self, t0, t1):
        s = [x for x in self.samples if t0 <= x[0] <= t1]
        if not s:  # very short run: take the nearest sample after t0
            s = [x for x in self.samples if x[0] >= t0][:1]
        return s

    def summarize(self, t0, t1):
        s = self.window(t0, t1)
        if not s:
            return {}
        rails = sorted({k for x in s for k in x[3]})
        dur = t1 - t0
        out = {"tegra_samples": len(s),
               "ram_peak_mb": max(x[1] for x in s if x[1] is not None),
               "gr3d_mean_pct": st.mean(x[2] for x in s if x[2] is not None)}
        for r in rails:
            mw = st.mean(x[3].get(r, 0) for x in s)
            out[f"{r}_mean_mW"] = mw
            out[f"{r}_energy_J"] = mw / 1000 * dur
        return out

    def ram_now(self):
        return self.samples[-1][1] if self.samples else None

    def stop(self):
        self.p.terminate()


# ---------------------------------------------------------------- records
def make_record(n_q, n_opt, image=None, qtype="choice"):
    state = {"ticket": {"subject": "Customer reports intermittent login failures",
                        "body": "Since yesterday the mobile app logs me out every few minutes. "
                                "I already reinstalled it. Order #A-1932, plan: business.",
                        "age_days": 2, "channel": "email"}}
    questions = {}
    for i in range(n_q):
        if qtype == "noul":
            questions[f"q{i}"] = {"type": "noul", "instructions": f"Statement {i}: is aspect {i} relevant to this ticket?"}
        else:
            questions[f"q{i}"] = {
                "type": "choice",
                "instructions": f"Question {i}: which category best describes aspect {i} of the ticket?",
                "criteria": {f"opt{j}": f"Category {j} for aspect {i}." for j in range(n_opt)},
            }
    rec = {"state": state, "questions": questions}
    if image is not None:
        rec["images"] = [image]
    return rec


def test_image(size):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(img)
    for k in range(0, size, max(8, size // 32)):
        d.line([(0, k), (size, size - k)], fill=(k % 255, 80, 160), width=2)
    d.text((10, 10), "ERROR 401: session expired", fill="black")
    return img


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Cloudflare/clef-flash")
    ap.add_argument("--questions", default="1,10,50,100")
    ap.add_argument("--options", default="2,4,8,16")
    ap.add_argument("--image", default="0", help="comma list of image sizes, 0 = no image")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--max-length", type=int, default=64000)
    ap.add_argument("--label", default="host-transformers")
    ap.add_argument("--out", default="results.jsonl")
    a = ap.parse_args()

    tg = Tegra()
    time.sleep(1.0)
    ram_before = tg.ram_now()

    path = snapshot_download(a.model)
    sys.path.insert(0, path)
    from joint_schema_model import collate_records, encode_record, load_release_model

    t0 = time.perf_counter()
    model, processor = load_release_model(path, device="cuda")
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    time.sleep(1.0)
    ram_loaded = tg.ram_now()
    weights_alloc = torch.cuda.memory_allocated()
    dev = torch.device("cuda")
    print(f"loaded {a.model} in {load_s:.1f}s  RAM {ram_before}->{ram_loaded} MB  "
          f"torch_alloc {weights_alloc/2**30:.2f} GiB", flush=True)

    meta = {"model": a.model, "label": a.label, "torch": torch.__version__,
            "nvpmodel": subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True).stdout.split("\n")[0],
            "load_s": load_s, "ram_before_mb": ram_before, "ram_loaded_mb": ram_loaded,
            "weights_alloc_gib": weights_alloc / 2**30}

    def forward(rec):
        enc = encode_record(processor.tokenizer, rec, processor=processor, max_length=a.max_length)
        batch = collate_records([enc], processor.tokenizer.pad_token_id, dev)
        with torch.inference_mode():
            logits = model(batch)[0]
        probs = [l.float().softmax(-1) for l in logits]
        torch.cuda.synchronize()
        return enc, probs

    for img_size in [int(x) for x in a.image.split(",")]:
        img = test_image(img_size) if img_size else None
        for n_q in [int(x) for x in a.questions.split(",")]:
            for n_opt in [int(x) for x in a.options.split(",")]:
                rec = make_record(n_q, n_opt, img)
                cfg = {**meta, "questions": n_q, "options": n_opt, "image": img_size}
                try:
                    for _ in range(a.warmup):
                        enc, _ = forward(rec)
                    n_tok = len(enc.input_ids) if hasattr(enc, "input_ids") else None
                    torch.cuda.reset_peak_memory_stats()
                    lat = []
                    t_start = time.perf_counter()
                    for _ in range(a.reps):
                        t = time.perf_counter(); forward(rec); lat.append(time.perf_counter() - t)
                    t_end = time.perf_counter()
                    time.sleep(0.25)  # let tegrastats catch the tail
                    lat_ms = sorted(x * 1000 for x in lat)
                    res = {**cfg, "status": "ok", "input_tokens": n_tok, "reps": a.reps,
                           "lat_mean_ms": st.mean(lat_ms), "lat_median_ms": st.median(lat_ms),
                           "lat_p95_ms": lat_ms[min(len(lat_ms) - 1, int(0.95 * len(lat_ms)))],
                           "lat_min_ms": lat_ms[0], "lat_per_question_ms": st.mean(lat_ms) / n_q,
                           "torch_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                           **tg.summarize(t_start, t_end)}
                    # AGX Orin: total module power = VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0
                    rails_j = [v for k, v in res.items() if k.endswith("_energy_J")]
                    rails_mw = [v for k, v in res.items() if k.endswith("_mean_mW")]
                    if rails_j:
                        res["total_mean_W"] = sum(rails_mw) / 1000
                        res["energy_per_req_J"] = sum(rails_j) / a.reps
                except torch.OutOfMemoryError as e:
                    res = {**cfg, "status": "oom", "error": str(e)[:300]}
                    torch.cuda.empty_cache()
                except Exception as e:
                    res = {**cfg, "status": "error", "error": f"{type(e).__name__}: {e}"[:300]}
                print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in res.items()
                                  if k not in meta or k == "label"}), flush=True)
                with open(a.out, "a") as f:
                    f.write(json.dumps(res) + "\n")
    tg.stop()


if __name__ == "__main__":
    main()
