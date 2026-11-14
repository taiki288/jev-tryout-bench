#!/usr/bin/env python3
"""Markdown tables from bench2 *.summary.jsonl files.

usage: summarize2.py results/*.summary.jsonl
"""
import json, sys


def f(v, nd=1):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:,.{nd}f}"
    return f"{v:,}" if isinstance(v, int) else str(v)


rows = [json.loads(l) for p in sys.argv[1:] for l in open(p)]
labels = list(dict.fromkeys(r["label"] for r in rows))

print("### 応答時間・メモリ・電力\n")
print("| 環境 | 画像 | 選択肢 | 入力tok | 平均 ms | p50 | p95 | 電力 W | J/req | GPU% | torch peak GiB | RAM peak MB |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for lab in labels:
    for r in sorted((r for r in rows if r["label"] == lab), key=lambda r: (r["image"], r["options"])):
        img = "なし" if r["image"] == 0 else f"{r['image']}px"
        if r["status"] != "ok":
            print(f"| {lab} | {img} | {r['options']} | **{r['status'].upper()}** | | | | | | | | |"); continue
        print(f"| {lab} | {img} | {r['options']} | {f(r['input_tokens_mean'], 0)} | {f(r['lat_mean_ms'])} | "
              f"{f(r['lat_p50_ms'])} | {f(r['lat_p95_ms'])} | {f(r['total_mean_W'])} | {f(r['energy_per_req_J'])} | "
              f"{f(r.get('gr3d_mean_pct'), 0)} | {f(r['torch_peak_gib'], 2)} | {f(r.get('ram_peak_mb'))} |")

print("\n### 応答のゆらぎ・正答率\n")
print("- 時間のゆらぎ: 同じ入力10回の標準偏差（質問10個の平均）と変動係数CV、全100回の標準偏差")
print("- 答えのゆらぎ: argmax一致率（同じ質問10回で最頻の答えと一致した割合）、確率の最大ぶれ（同じ質問・同じ選択肢での max-min の最大値）")
print("- 正答率: 正解ラベル（各質問に1つ）を argmax で選べた割合。量子化比較用の参考値\n")
print("| 環境 | 画像 | 選択肢 | 時間std ms（同一入力） | CV | std ms（全体） | argmax一致率 | 確率の最大ぶれ | 正答率 |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
for lab in labels:
    for r in sorted((r for r in rows if r["label"] == lab and r["status"] == "ok"), key=lambda r: (r["image"], r["options"])):
        img = "なし" if r["image"] == 0 else f"{r['image']}px"
        print(f"| {lab} | {img} | {r['options']} | {f(r['jitter_within_std_ms'], 2)} | {r['jitter_within_cv']*100:.2f}% | "
              f"{f(r['lat_std_overall_ms'], 1)} | {r['argmax_agree']*100:.0f}% | {r['prob_spread_max']:.2e} | "
              f"{r['accuracy']*100:.0f}% |")
