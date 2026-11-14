#!/usr/bin/env python3
"""Export the exact inputs used by bench2.py and the model's answers.

usage: export_dataset.py OUTDIR results/*.runs.jsonl

OUTDIR/
  images/test_<size>.png        images fed to the model
  questions.json                10 tickets + gold label
  options_K<k>.json             per question: option id -> label, gold id (exactly as fed)
  records_K<k>_img<size>.json   full input record (state + question + criteria), image as file name
  answers.csv                   one row per (env, K, image, question): answer per rep, top-3, correctness
"""
import csv, json, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench2 import LABELS, TICKETS, make_record  # noqa: E402
from bench import test_image  # noqa: E402

out, run_files = sys.argv[1], sys.argv[2:]
os.makedirs(f"{out}/images", exist_ok=True)

runs = [json.loads(l) for p in run_files for l in open(p)]
ks = sorted({r["options"] for r in runs})
imgs = sorted({r["image"] for r in runs})

for s in imgs:
    if s:
        test_image(s).save(f"{out}/images/test_{s}.png")

json.dump({"labels_pool": LABELS,
           "tickets": [{"question": i, "body": t, "gold_label": g} for i, (t, g) in enumerate(TICKETS)],
           "instructions": "Which topic / kind best describes this support ticket?"},
          open(f"{out}/questions.json", "w"), ensure_ascii=False, indent=1)

crit = {}
for k in ks:
    opts = []
    for qi in range(len(TICKETS)):
        rec, gold = make_record(qi, k)
        c = rec["questions"]["route"]["criteria"]
        crit[(k, qi)] = c
        opts.append({"question": qi, "gold_id": gold, "gold_label": c[gold], "criteria": c})
    json.dump(opts, open(f"{out}/options_K{k}.json", "w"), ensure_ascii=False, indent=1)
    for s in imgs:
        recs = []
        for qi in range(len(TICKETS)):
            rec, _ = make_record(qi, k)
            if s:
                rec["images"] = [f"images/test_{s}.png"]
            recs.append(rec)
        json.dump(recs, open(f"{out}/records_K{k}_img{s}.json", "w"), ensure_ascii=False, indent=1)

groups = defaultdict(list)
for r in runs:
    groups[(r["label"], r["options"], r["image"], r["question"])].append(r)

with open(f"{out}/answers.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["env", "options", "image", "question", "gold_label", "answer_label", "correct",
                "answers_all_reps_same", "p_gold", "top1", "top2", "top3", "mean_lat_ms"])
    for (lab, k, img, qi), rs in sorted(groups.items()):
        c = crit[(k, qi)]
        ids = sorted(c)  # option_ids order == criteria key order in encode_record (o000..)
        ids = list(c.keys())
        r0 = rs[0]
        top = sorted(range(len(r0["probs"])), key=lambda j: -r0["probs"][j])[:3]
        w.writerow([lab, k, img, qi, c[r0["gold"]], c[r0["argmax"]], r0["correct"],
                    len({r["argmax"] for r in rs}) == 1, f"{r0['p_gold']:.4f}",
                    *[f"{c[ids[j]]} ({r0['probs'][j]:.3f})" for j in top],
                    f"{sum(r['lat_ms'] for r in rs) / len(rs):.1f}"])
print("wrote", out)
