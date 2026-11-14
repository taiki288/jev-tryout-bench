#!/usr/bin/env python3
"""Clef benchmark per the lab spec.

For each option count K (default 10/50/100) and each image setting:
  - 10 distinct questions (different tickets), each a single `choice` question with K options
    (the correct label + K-1 distractors drawn from a fixed pool of 100 labels)
  - each question is answered `--reps` times (default 10)
Records per run: latency, argmax, correctness, full probability vector.
Summarises per (K, image): mean/p50/p95 latency, jitter (std, CV), answer stability
(argmax agreement, max prob spread across reps), accuracy, memory and power.

Outputs:
  <out>.runs.jsonl     one line per run (raw)
  <out>.summary.jsonl  one line per (K, image)
"""
import argparse, json, random, statistics as st, subprocess, sys, time

import torch
from huggingface_hub import snapshot_download

# gptqmodel patches triton's Autotuner on import. Autotuners created earlier (e.g. by
# flash-linear-attention, used by Clef / Clef-flash) then lack its `_cache_lock` attribute and
# crash at run time, so import it first when it is installed.
try:
    import gptqmodel  # noqa: F401
except ImportError:
    pass

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from bench import Tegra, test_image  # noqa: E402

TOPICS = ["account login", "password reset", "billing", "refund", "shipping", "delivery delay",
          "product defect", "subscription", "cancellation", "data privacy", "security incident",
          "API integration", "mobile app", "website outage", "invoice", "tax documents",
          "plan upgrade", "plan downgrade", "gift card", "warranty"]
KINDS = ["general question", "complaint", "urgent issue", "change request", "feedback"]
LABELS = [f"{t} / {k}" for t in TOPICS for k in KINDS]  # 100 labels

TICKETS = [
    ("I can't log in since this morning, the app says my session expired every time. I have a client demo in an hour!",
     "account login / urgent issue"),
    ("You charged me twice for October. Please give the second payment back.", "refund / change request"),
    ("My package was supposed to arrive last Tuesday and tracking hasn't moved for 6 days. This is unacceptable.",
     "delivery delay / complaint"),
    ("Does the business plan include SSO, or do I need to upgrade to enterprise for that?",
     "plan upgrade / general question"),
    ("The blender I received makes a burning smell after 10 seconds of use.", "product defect / complaint"),
    ("Please delete all personal data you hold about me under GDPR.", "data privacy / change request"),
    ("Our webhook calls to /v2/orders started returning 500 errors in production 20 minutes ago.",
     "API integration / urgent issue"),
    ("Just wanted to say the new dark mode in the iOS app looks great, thanks!", "mobile app / feedback"),
    ("Where can I download the invoice for my September purchase? Our accountant needs it.",
     "invoice / general question"),
    ("I got an email saying someone logged into my account from another country. I didn't do that.",
     "security incident / urgent issue"),
]


def make_record(ticket_idx, k, image=None, seed=0):
    text, gold = TICKETS[ticket_idx]
    rng = random.Random(f"{seed}-{ticket_idx}-{k}")
    distractors = rng.sample([l for l in LABELS if l != gold], k - 1)
    opts = distractors + [gold]
    rng.shuffle(opts)
    criteria = {f"o{j:03d}": lab for j, lab in enumerate(opts)}
    gold_id = next(i for i, lab in criteria.items() if lab == gold)
    rec = {"state": {"ticket": {"id": f"T-{1000 + ticket_idx}", "channel": "email", "body": text}},
           "questions": {"route": {"type": "choice",
                                   "instructions": "Which topic / kind best describes this support ticket?",
                                   "criteria": criteria}}}
    if image is not None:
        rec["images"] = [image]
    return rec, gold_id


def nvpmodel_mode():
    """Power mode; inside a container nvpmodel is unavailable, so the host passes it via NVPMODEL_Q."""
    import os
    try:
        return subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True).stdout.split("\n")[0]
    except FileNotFoundError:
        return os.environ.get("NVPMODEL_Q", "unknown")


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Cloudflare/clef-flash", help="weights repo")
    ap.add_argument("--code-repo", default=None,
                    help="official Cloudflare repo to import joint_schema_model.py from (default: --model if it is a Cloudflare repo)")
    ap.add_argument("--options", default="10,50,100")
    ap.add_argument("--image", default="0,1024", help="comma list of image sizes, 0 = no image")
    ap.add_argument("--bnb", choices=["int8", "nf4"], default=None,
                    help="quantize the (bf16) weights at load time with bitsandbytes")
    ap.add_argument("--n-questions", type=int, default=10)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=64000)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True, help="output prefix")
    a = ap.parse_args()

    tg = Tegra(); time.sleep(1.0)
    ram_before = tg.ram_now()
    path = snapshot_download(a.model)
    code_repo = a.code_repo or a.model
    assert code_repo.startswith("Cloudflare/"), "loader code must come from an official Cloudflare repo (--code-repo)"
    sys.path.insert(0, snapshot_download(code_repo))
    from joint_schema_model import collate_records, encode_record, load_release_model
    print(f"loader code from {code_repo} | weights from {a.model}", flush=True)

    t0 = time.perf_counter()
    kw = {}
    if a.bnb:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = (BitsAndBytesConfig(load_in_8bit=True) if a.bnb == "int8" else
                                     BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                        bnb_4bit_compute_dtype=torch.bfloat16))
    model, processor = load_release_model(path, device="cuda", **kw)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    time.sleep(1.0)
    dev = torch.device("cuda")
    import transformers
    meta = {"model": a.model, "label": a.label, "torch": torch.__version__, "transformers": transformers.__version__,
            "bnb": a.bnb,
            "nvpmodel": nvpmodel_mode(),
            "load_s": load_s, "ram_before_mb": ram_before, "ram_loaded_mb": tg.ram_now(),
            "weights_alloc_gib": torch.cuda.memory_allocated() / 2**30}
    print(f"loaded {a.model} in {load_s:.1f}s  RAM {meta['ram_before_mb']}->{meta['ram_loaded_mb']} MB", flush=True)

    def run(rec):
        t = time.perf_counter()
        enc = encode_record(processor.tokenizer, rec, processor=processor, max_length=a.max_length)
        batch = collate_records([enc], processor.tokenizer.pad_token_id, dev)
        with torch.inference_mode():
            logits = model(batch)[0]
        probs = logits[0].float().softmax(-1).cpu()
        torch.cuda.synchronize()
        return time.perf_counter() - t, enc, probs

    runs_f = open(a.out + ".runs.jsonl", "a")
    sum_f = open(a.out + ".summary.jsonl", "a")
    for img_size in [int(x) for x in a.image.split(",")]:
        img = test_image(img_size) if img_size else None
        for k in [int(x) for x in a.options.split(",")]:
            cfg = {**meta, "options": k, "image": img_size}
            try:
                recs = [make_record(i, k, img) for i in range(a.n_questions)]
                for i in range(a.warmup):
                    run(recs[i % len(recs)][0])
                torch.cuda.reset_peak_memory_stats()
                per_q, all_lat = [], []
                t_start = time.perf_counter()
                for qi, (rec, gold_id) in enumerate(recs):
                    lat, probs_list, argmaxes, n_tok = [], [], [], None
                    for r in range(a.reps):
                        dt, enc, probs = run(rec)
                        option_ids = list(enc.questions[0].option_ids)
                        am = option_ids[int(probs.argmax())]
                        lat.append(dt * 1000); probs_list.append(probs); argmaxes.append(am)
                        n_tok = len(enc.input_ids)
                        runs_f.write(json.dumps({"label": a.label, "model": a.model, "options": k, "image": img_size,
                                                 "question": qi, "rep": r, "lat_ms": dt * 1000, "argmax": am,
                                                 "gold": gold_id, "correct": am == gold_id,
                                                 "p_gold": float(probs[option_ids.index(gold_id)]),
                                                 "probs": [round(float(p), 6) for p in probs]}) + "\n")
                    P = torch.stack(probs_list)
                    mode = max(set(argmaxes), key=argmaxes.count)
                    per_q.append({"question": qi, "input_tokens": n_tok, "lat_mean_ms": st.mean(lat),
                                  "lat_std_ms": st.pstdev(lat), "lat_cv": st.pstdev(lat) / st.mean(lat),
                                  "argmax_agree": argmaxes.count(mode) / len(argmaxes),
                                  "prob_spread_max": float((P.max(0).values - P.min(0).values).max()),
                                  "correct_rate": sum(x == gold_id for x in argmaxes) / len(argmaxes)})
                    all_lat += lat
                    print(f"  K={k} img={img_size} q{qi}: {st.mean(lat):.0f}±{st.pstdev(lat):.1f} ms "
                          f"tok={n_tok} agree={per_q[-1]['argmax_agree']:.2f} correct={per_q[-1]['correct_rate']:.1f}",
                          flush=True)
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
                       # jitter: within-question (same input repeated) vs. overall
                       "jitter_within_std_ms": st.mean(q["lat_std_ms"] for q in per_q),
                       "jitter_within_cv": st.mean(q["lat_cv"] for q in per_q),
                       "lat_std_overall_ms": st.pstdev(all_lat),
                       # answer stability / quality
                       "argmax_agree": st.mean(q["argmax_agree"] for q in per_q),
                       "prob_spread_max": max(q["prob_spread_max"] for q in per_q),
                       "accuracy": st.mean(q["correct_rate"] for q in per_q),
                       "torch_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                       "total_mean_W": sum(rails_mw) / 1000 if rails_mw else None,
                       "energy_per_req_J": sum(rails_j) / n if rails_j else None,
                       **tsum, "per_question": per_q}
            except torch.OutOfMemoryError as e:
                res = {**cfg, "status": "oom", "error": str(e)[:300]}; torch.cuda.empty_cache()
            except Exception as e:
                res = {**cfg, "status": "error", "error": f"{type(e).__name__}: {e}"[:300]}
            print(json.dumps({kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in res.items()
                              if kk not in meta and kk != "per_question"}), flush=True)
            sum_f.write(json.dumps(res) + "\n"); sum_f.flush(); runs_f.flush()
    tg.stop()


if __name__ == "__main__":
    main()
